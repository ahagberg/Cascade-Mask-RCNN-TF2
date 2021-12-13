import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from model import anchors
from model.losses import MaskRCNNLoss, FastRCNNLoss, RPNLoss
from model.models.fpn import FPNNetwork
from models.heads import RPNHead, BoxHead, MaskHead
from model.models.resnet50 import ResNet50
from ops import roi_ops, spatial_transform_ops, postprocess_ops, training_ops
from model.models.swin_pretrained import swin_transformer

class MaskRCNN(tf.keras.Model):

    def __init__(self, params, name='mrcnn', backbone_model='swin_base_384', trainable=True, rgbd=False,
        *args, **kwargs):
        super().__init__(name=name, trainable=trainable, *args, **kwargs)
        self._params = params

        if rgbd:
            self.depth_conv = Conv2D(3, (1,1))
        else:
            self.depth_conv = None

        if 'swin' in backbone_model:
            self.backbone = swin_transformer(use_tpu=params.use_tpu, model_name=backbone_model)
        elif 'resnet' in backbone_model.lower():
            self.backbone = ResNet50()
        else:
            raise NotImplementedError(f"Backbone {backbone_model} is not implemented")
        

        self.fpn = FPNNetwork(
            min_level=self._params.min_level,
            max_level=self._params.max_level,
            trainable=trainable
        )

        self.rpn_head = RPNHead(
            name="rpn_head",
            num_anchors=len(self._params.aspect_ratios * self._params.num_scales),
            trainable=trainable
        )

        self.box_heads = [
            BoxHead(
                num_classes=self._params.num_classes,
                mlp_head_dim=self._params.fast_rcnn_mlp_head_dim,
                trainable=trainable
            )
        ]

        self.mask_heads = [
            MaskHead(
                num_classes=self._params.num_classes,
                mrcnn_resolution=self._params.mrcnn_resolution,
                trainable=trainable,
                name="mask_head"
            )
        ]

        self.mask_rcnn_losses = [ MaskRCNNLoss() ]

        self.fast_rcnn_losses = [
            FastRCNNLoss(
                num_classes=self._params.num_classes
            )
        ]

        self.rpn_loss = RPNLoss(
            batch_size=self._params.train_batch_size,
            rpn_batch_size_per_im=self._params.rpn_batch_size_per_im,
            min_level=self._params.min_level,
            max_level=self._params.max_level
        )


    



    def detection_stage(self, fpn_features, current_rois, use_mask, 
                        training, model_inputs, rpn_outputs=None,
                        stage=0, nms_threshold=0.5):
        outputs = {}
        outputs.update(rpn_outputs)
        # run frcnn head
        # Performs multi-level RoIAlign.
        if training:
            current_rois = tf.stop_gradient(current_rois)
            
            # Sampling
            box_targets, class_targets, current_rois, proposal_to_label_map = training_ops.proposal_label_op(
                current_rois,
                outputs['gt_boxes'],
                outputs['gt_classes'],
                batch_size_per_im=self._params.batch_size_per_im,
                fg_fraction=self._params.fg_fraction,
                fg_thresh=nms_threshold,
                bg_thresh_hi=nms_threshold,
                bg_thresh_lo=self._params.bg_thresh_lo
            )
            
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_features,
            boxes=current_rois,
            output_size=7,
            training=training
        )


        class_outputs, box_outputs, _ = self.box_heads[stage](inputs=box_roi_features)
        correct_classes = class_targets if training else tf.argmax(class_outputs, axis=-1)
        
        new_rois = roi_ops.box_outputs_to_rois(box_outputs, current_rois,
                    correct_classes, outputs['image_info'],
                    self._params.bbox_reg_weights[stage])


        if not training:
            if self._params.use_tpu:
                generate_fn = postprocess_ops.generate_detections_tpu
            else:
                generate_fn = postprocess_ops.generate_detections_gpu
            detections = generate_fn(
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=current_rois,
                image_info=model_inputs['image_info'],
                pre_nms_num_detections=self._params.test_rpn_post_nms_topn,
                post_nms_num_detections=self._params.test_detections_per_image,
                nms_threshold=nms_threshold,
                bbox_reg_weights=self._params.bbox_reg_weights[stage]
            )
            
            outputs.update({
                'num_detections': detections[0],
                'detection_boxes': detections[1],
                'detection_classes': detections[2],
                'detection_scores': detections[3],
            })

        else:  # is training
            encoded_box_targets = training_ops.encode_box_targets(
                boxes=current_rois,
                gt_boxes=box_targets,
                gt_labels=class_targets,
                bbox_reg_weights=self._params.bbox_reg_weights[stage]
            )
            
            outputs.update({
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'box_targets': encoded_box_targets,
                'class_targets': class_targets,
            })

          
        #outputs.update(rpn_outputs)


        # Faster-RCNN mode.
        if not use_mask:
            if training:
                self._add_detection_losses(outputs, use_mask, stage)
            return new_rois, outputs

        # Mask sampling
        if not training:
            selected_box_rois = outputs['detection_boxes']
            class_indices = outputs['detection_classes']

        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=current_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=int(self._params.batch_size_per_im * self._params.fg_fraction)
            )

            class_indices = tf.cast(selected_class_targets, dtype=tf.int32)

        mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_features,
            boxes=selected_box_rois,
            output_size=14,
            training=training
        )

        mask_outputs = self.mask_heads[stage](
            inputs=(mask_roi_features, class_indices),
            training=training
        )

        if training:
            mask_targets = training_ops.get_mask_targets(

                fg_boxes=selected_box_rois,
                fg_proposal_to_label_map=proposal_to_label_map,
                fg_box_targets=selected_box_targets,
                mask_gt_labels=model_inputs['cropped_gt_masks'],
                output_size=self._params.mrcnn_resolution
            )

            outputs.update({
                'mask_outputs': mask_outputs,
                'mask_targets': mask_targets,
                'selected_class_targets': selected_class_targets,
            })
            self._add_detection_losses(outputs, use_mask, stage)

        else:
            outputs.update({
                'detection_masks': tf.nn.sigmoid(mask_outputs),
            })
        
        return new_rois, outputs


    def rpn_head_fn(self, features, min_level=2, max_level=6, training=None):
        """Region Proposal Network (RPN) for Mask-RCNN."""
        scores_outputs = dict()
        box_outputs = dict()

        for level in range(min_level, max_level + 1):
            scores_outputs[level], box_outputs[level] = self.rpn_head(features[level], training=training)

        return scores_outputs, box_outputs


    def rpn_network(self, fpn_feats, all_anchors, inputs, training):
        rpn_score_outputs, rpn_box_outputs = self.rpn_head_fn(
            features=fpn_feats,
            min_level=self._params.min_level,
            max_level=self._params.max_level
        )

        outputs = dict(inputs)

        outputs.update({'fpn_features': fpn_feats})


        if training:
            rpn_pre_nms_topn = self._params.train_rpn_pre_nms_topn
            rpn_post_nms_topn = self._params.train_rpn_post_nms_topn
            rpn_nms_threshold = self._params.train_rpn_nms_threshold

        else:
            rpn_pre_nms_topn = self._params.test_rpn_pre_nms_topn
            rpn_post_nms_topn = self._params.test_rpn_post_nms_topn
            rpn_nms_threshold = self._params.test_rpn_nms_thresh

        rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            all_anchors=all_anchors,
            image_info=inputs['image_info'],
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=self._params.rpn_min_size,
            bbox_reg_weights=None, use_tpu=self._params.use_tpu
        )

        rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)
        

        outputs.update(
            {
                'rpn_score_outputs': rpn_score_outputs,
                'rpn_box_outputs': rpn_box_outputs,
            }
        )

        if training:
            
            self._add_rpn_losses(outputs)
            
        outputs['box_rois'] = rpn_box_rois

        return outputs

    def get_inputs(self, inputs, training):

        batch_size, image_height, image_width, _ = inputs['images'].get_shape().as_list()

        if 'source_ids' not in inputs:
            inputs['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)

        all_anchors = anchors.Anchors(self._params.min_level, self._params.max_level,
                                      self._params.num_scales, self._params.aspect_ratios,
                                      self._params.anchor_scale,
                                      (image_height, image_width))

        if self.depth_conv:
            processed_depth = self.depth_conv(inputs['images'][:,:,:,3:])
            backbone_input = processed_depth + inputs['images'][:,:,:,:3] 
        else:
            backbone_input = inputs['images']

        backbone_feats = self.backbone(backbone_input)
        fpn_feats = self.fpn(backbone_feats, training=training)
        rpn_outputs = self.rpn_network(fpn_feats, all_anchors, inputs, training)
        return rpn_outputs, fpn_feats


    def filter_outputs(self, outputs):
        model_outputs = [
            'source_ids', 'image_info',
            'num_detections', 'detection_boxes',
            'detection_classes', 'detection_scores',
            'detection_masks'
        ]
        return {
            name: tf.identity(tensor, name=name)
            for name, tensor in outputs.items()
            if name in model_outputs
        }


    def call(self, inputs, training=None, mask=None):
        
        outputs, fpn_feats = self.get_inputs(inputs, training)

        _, detection_outputs = self.detection_stage(
            fpn_feats, 
            current_rois=outputs['box_rois'], 
            use_mask=True,
            training=training, 
            model_inputs=inputs, 
            rpn_outputs=outputs
        )

        outputs.update(detection_outputs)
        
        return self.filter_outputs(outputs)
        

    def _add_detection_losses(self, model_outputs, use_mask, stage=0):
        if use_mask:
            mask_rcnn_loss = self.mask_rcnn_losses[stage](model_outputs)
            mask_rcnn_loss *= self._params.mrcnn_weight_loss_mask
            self.add_loss(mask_rcnn_loss)
            self.add_metric(mask_rcnn_loss, name=f'mask_rcnn_loss_{stage}')

        fast_rcnn_class_loss, fast_rcnn_box_loss = self.fast_rcnn_losses[stage](model_outputs)
        fast_rcnn_box_loss *= self._params.fast_rcnn_box_loss_weight
        self.add_loss(fast_rcnn_box_loss)
        self.add_metric(fast_rcnn_box_loss, name=f'fast_rcnn_box_loss_{stage}')
        self.add_loss(fast_rcnn_class_loss)
        self.add_metric(fast_rcnn_class_loss, name=f'fast_rcnn_class_loss_{stage}')



    def _add_rpn_losses(self, model_outputs):
        rpn_score_loss, rpn_box_loss = self.rpn_loss(model_outputs)
        rpn_box_loss *= self._params.rpn_box_loss_weight
        self.add_loss(rpn_box_loss)
        self.add_metric(rpn_box_loss, name='rpn_box_loss')
        self.add_loss(rpn_score_loss)
        self.add_metric(rpn_score_loss, name='rpn_score_loss')
        


    def get_config(self):
        pass
