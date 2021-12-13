
import numpy as np
import h5py
import tensorflow as tf
from tensorflow._api.v2 import image
from dataset.dataset_parser import dataset_parser
import pickle
import os
from tqdm.auto import tqdm


def add_depth_to_image(img, depth): # Is the name suitable?
    """
    Concatenates image with the depth info
    """

    img_height = depth.shape[0]
    img_width = depth.shape[1]
    target_height = img.shape[0]
    target_width = img.shape[1]
    scale_if_resize_height = target_height / img_height
    scale_if_resize_width = target_width / img_width
    scale = min(scale_if_resize_height, scale_if_resize_width)
    scaled_height = round(scale * img_height)
    scaled_width = round(scale * img_width)
    expanded_depth = tf.expand_dims(depth, axis=-1)
    scaled_depth = tf.image.resize(expanded_depth, [scaled_height, scaled_width], method=tf.image.ResizeMethod.BILINEAR)   
    padded_depth = tf.image.pad_to_bounding_box(scaled_depth, 0, 0, target_height, target_width)
    return tf.concat([img, padded_depth], axis=-1)


def create_nyu_dataset(dir, conf, train=True, use_depth=True):
    file_name = "nyu_dataset_train.pickle" if train else "nyu_dataset_test.pickle"
    file_name = 'depth' + file_name if use_depth else file_name
    file_path = os.path.join(conf.data_dir, file_name)
    print("file path ", file_path)
    if os.path.exists(file_path):
        print(f"Reading data from file {file_path} ...")
        with open(file_path, 'rb') as f:
            data_features = pickle.load(f)
    else:
        f = h5py.File(dir)
        print('Reading dataset ...')
        images = np.array(f['images']).transpose((0, 3, 2, 1))               
        instances = np.array(f['instances']).transpose((0, 2, 1))
        labels = np.array(f['labels']).transpose((0, 2, 1))
        names_array = np.array(f['names'])
        
        if use_depth:
            depths = np.array(f['depths']).transpose((0, 2, 1))
            depth_mean = np.mean(depths)
            depth_stdev = np.std(depths)
            depths = (depths - depth_mean) / depth_stdev
        
        if isinstance(conf.USE_CLASSES, list):
            label_mapping = np.zeros((names_array.shape[-1] + 1,))
            for i in range(names_array.shape[-1]):
                obj = f[names_array[0][i]]
                class_name = ''.join(chr(i) for i in obj[:,0])
                if class_name in conf.USE_CLASSES:
                    label_mapping[i+1] = conf.USE_CLASSES.index(class_name) + 1
            
            labels = label_mapping[labels]

        n_samples = len(images)

        n_train = round(n_samples * 0.8)
        start = 0 if train else n_train
        end = n_train if train else n_samples

        print('Generating input data ...')
        for i in tqdm(range(start, end)):
            mask, class_ids = generate_mask(labels[i], instances[i], conf)
            gt_area = np.sum(mask, axis=(0,1))
            boxes = extract_bboxes(mask)
            is_crowd = [False] * len(gt_area)

            
            
            data = {'image': images[i], 'source_id' : tf.constant(i), 
                    'height': images.shape[1], 'width': images.shape[2],
                    'groundtruth_classes': class_ids , 'groundtruth_is_crowd': is_crowd,
                    'groundtruth_area': gt_area , 'groundtruth_boxes': boxes,
                    'groundtruth_instance_masks': tf.transpose(mask, perm=[2, 0, 1]), 
                    'groundtruth_instance_masks_png': mask }
            
            input_features, output_features = dataset_parser(data,
                            mode='train', params=conf, 
                            use_instance_mask = conf.include_mask,seed=conf.seed)

            if use_depth:
                depth = depths[i]
                img = input_features['images']
                input_features['images'] = add_depth_to_image(input_features['images'], depth)
                

            if i == start:
                feature_dict = {}
                for key, val in input_features.items():
                    feature_dict[key] = [val]
            else:
                for key, val in input_features.items():
                    feature_dict[key].append(val)
        data_features = (feature_dict, output_features)
        print(f"Saving data to file {file_path} ...")
        with open(file_path, 'wb') as f:
            pickle.dump(data_features, f)

    print('Creating dataset ...')

    dataset = tf.data.Dataset.from_tensor_slices(data_features)
    return dataset

#########################################
#
#########################################
def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.
    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    #print('active class ids in compose: ', active_class_ids)
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

    
def load_image_gt(image, labels, instance_numbers, config, image_id, use_mini_mask=False):
    mask, class_ids = generate_mask(labels, instance_numbers, config)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    # extract_bboxes：根据mask生成box bounding
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([config.NUM_CLASSES], dtype=np.int32)
    active_class_ids[labels.astype('int32')] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)
    return image, image_meta, class_ids, bbox, mask



def generate_mask(labels, instance_numbers, config):
    H, W = labels.shape
    instance_class = np.where(labels > 0, labels * (config.MAX_INSTANCE_NUM + 1) + instance_numbers, 0)
    class_ids = np.unique(instance_class) // (config.MAX_INSTANCE_NUM + 1)
    unique_instances = np.unique(instance_class)
    mask = np.zeros((H, W, len(unique_instances) - 1), dtype='int8')
    for i, instance in enumerate(unique_instances[1:]):
        mask[:,:,i] = np.where(instance_class==instance, 1, 0)
    return mask, class_ids[1:]

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.float32)