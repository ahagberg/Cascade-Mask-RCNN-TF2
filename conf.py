import tensorflow as tf
class Conf(object):
    USE_CLASSES = ['drawer',
        'dresser',
        'shoe',
        'stuffed animal',
        'television',
        'person',
        'candlestick',
        'tissue box',
        'jar',
        'plant',
        'vase',
        'electrical outlet',
        'night stand',
        'monitor',
        'garbage bin',
        'faucet',
        'floor mat',
        'sink',
        'mirror',
        'curtain',
        'towel',
        'bowl',
        'bookshelf',
        'blinds',
        'desk',
        'bed',
        'cup',
        'shelves',
        'counter',
        'clothes',
        'lamp',
        'sofa',
        'light',
        'bag',
        'door',
        'window',
        'box',
        'table',
        'book',
        'paper',
        'books',
        'bottle',
        'pillow',
        'cabinet',
        'chair',
        'picture']
    # input preFprocessing parameters
    image_size= (384, 384) #(832, 1344)
    augment_input_data=True
    gt_mask_size=112

    # dataset specific parameters
    num_classes= 1 + len(USE_CLASSES)
    skip_crowd_during_training=True
    use_category=True

    # Region Proposal Network
    rpn_positive_overlap=0.7
    rpn_negative_overlap=0.3
    rpn_batch_size_per_im=256
    rpn_fg_fraction=0.5
    rpn_min_size=0.

    # Proposal layer.
    batch_size_per_im=512
    fg_fraction=0.25
    fg_thresh=0.5
    bg_thresh_hi=0.5
    bg_thresh_lo=0.

    # Faster-RCNN heads.
    fast_rcnn_mlp_head_dim=1024
    bbox_reg_weights=[
        [10.0, 10.0, 5.0, 5.0],
        [20.0, 20.0, 10.0, 10.0],
        [30.0, 30.0, 15.0, 15.0],
    ]

    # Mask-RCNN heads.
    include_mask=True  # whether or not to include mask branch.   # ===== Not existing in MLPerf ===== #
    mrcnn_resolution=28

    # training
    train_rpn_pre_nms_topn=2000
    train_rpn_post_nms_topn=1000
    train_rpn_nms_threshold=0.7

    # evaluation
    test_detections_per_image=100
    test_nms=0.5
    test_rpn_pre_nms_topn=1000
    test_rpn_post_nms_topn=1000
    test_rpn_nms_thresh=0.7

    # model architecture
    min_level=2
    max_level=6
    num_scales=1
    aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scale=8.0
    stage_nms_thresholds=[0.5, 0.6, 0.7]
    mask_stages=[False, False, True]

    # localization loss
    rpn_box_loss_weight=1.0
    fast_rcnn_box_loss_weight=1.0
    mrcnn_weight_loss_mask=1.0

    # other
    checkpoint_name_format='nvidia_mrcnn_tf2.ckpt'

    train_batch_size = 2
    mode = ['train', 'eval']
    MAX_INSTANCE_NUM = 60
    #max_num_instances = MAX_INSTANCE_NUM 
    
    data_dir = '/data'
    model_dir = '/results'
    backbone_checkpoint = ''
    eval_file = ''
    epochs = 20
    seed = None
    use_tpu = True
    amp = False
    xla = False
    steps_per_epoch = 0
    init_learning_rate = 0.0
    l2_weight_decay = 1e-4
    learning_rate_values = [1e-2, 1e-3, 1e-4]
    learning_rate_boundaries = [0.3, 8.0, 10.0]
    momentum = 0.9
    finetune_bn = False
    use_synthetic_data = False
    log_every = 100
    log_file = 'mrcnn-dll.json'
    log_warmup_steps = 100
    log_graph = False
    log_tensorboard = ''
    eagerly = tf.executing_eagerly()
    train_size = 0 
    eval_batch_size = 1
    verbose = 0