import os, logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

def load_tf_records(filepath):
    filenames = tf.io.gfile.glob(filepath)
    
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = True
    
    dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=tf.data.experimental.AUTOTUNE).with_options(ignore_order)
    return dataset

def tf_records_file_features_description_train():
    image_feature_description = {
        
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([],tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'label/P': tf.io.FixedLenFeature([], tf.float32),
        'label/K': tf.io.FixedLenFeature([], tf.float32),
        'label/Mg': tf.io.FixedLenFeature([], tf.float32),
        'label/Ph': tf.io.FixedLenFeature([], tf.float32),
    }
    return image_feature_description

def decode_dataset_train_val(example_proto):
    features=tf.io.parse_single_example(example_proto, tf_records_file_features_description_train())

    patch=features['image']
    height=features['image/height']
    width=features['image/width']
    patch=tf.io.decode_raw(patch,tf.int16)
    patch=tf.reshape(patch,[height,width,150])
    filename=features['image/filename']

    P=features['label/P']
    K=features['label/K']
    Mg=features['label/Mg']
    Ph=features['label/Ph']

    height=tf.cast(features['image/height'],tf.int32)
    width=tf.cast(features['image/width'],tf.int32)

    label=[P,K,Mg,Ph]

    return patch, label, height, width

def tf_records_file_features_description_test():
    image_feature_description = {
        
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([],tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),

    }
    return image_feature_description
    
    
def decode_dataset_test(example_proto):
    features=tf.io.parse_single_example(example_proto, tf_records_file_features_description_test())

    patch=features['image']
    height=features['image/height']
    width=features['image/width']
    patch=tf.io.decode_raw(patch,tf.int16)
    patch=tf.reshape(patch,[height,width,150])
    filename=features['image/filename']

    height=features['image/height']
    width=features['image/width']

    return filename, patch, height, width