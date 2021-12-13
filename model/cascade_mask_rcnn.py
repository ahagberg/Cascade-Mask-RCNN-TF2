from .mask_rcnn import MaskRCNN
from model.models.heads import BoxHead, MaskHead
from model.losses import MaskRCNNLoss, FastRCNNLoss

class CascadeMaskRCNN(MaskRCNN):
    def __init__(self, params, name='cascade-mrcnn', backbone_model='swin_base_384', trainable=True, rgbd=False,
        *args, **kwargs):
        super().__init__(params, name=name, backbone_model=backbone_model, trainable=trainable, rgbd=rgbd, *args, **kwargs)

        for i in range(2):
            self.box_heads.append(
                BoxHead(
                    num_classes=self._params.num_classes,
                    mlp_head_dim=self._params.fast_rcnn_mlp_head_dim,
                    trainable=trainable,
                    name=f'box_head_{i}'
                )
            )

            self.mask_heads.append(
                MaskHead(
                    num_classes=self._params.num_classes,
                    mrcnn_resolution=self._params.mrcnn_resolution,
                    trainable=trainable,
                    name=f"mask_head_{i}"
                )
            )

            self.fast_rcnn_losses.append(
                FastRCNNLoss(
                    num_classes=self._params.num_classes
                )
            )
        
        self.mask_rcnn_losses = []
        for i, use_mask in enumerate(self._params.mask_stages):
            if not use_mask:
                self.mask_heads[i] = None # Remove mask head if not used
                self.mask_rcnn_losses.append(None)
            else:
                self.mask_rcnn_losses.append(MaskRCNNLoss())

        



    def call(self, inputs, training=None, mask=None):
        outputs, fpn_feats = self.get_inputs(inputs, training)

        stage_boxes = outputs['box_rois']

        for stage in range(3):
            stage_boxes, detection_outputs = self.detection_stage(
                fpn_feats, 
                current_rois=stage_boxes, 
                use_mask=self._params.mask_stages[stage],
                training=training, 
                model_inputs=inputs, 
                rpn_outputs=outputs,
                stage=stage,
                nms_threshold=self._params.stage_nms_thresholds[stage]
            )
            outputs.update(detection_outputs)

        return self.filter_outputs(outputs)