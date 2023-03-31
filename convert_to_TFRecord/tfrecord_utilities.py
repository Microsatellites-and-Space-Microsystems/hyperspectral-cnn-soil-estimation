import os, logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
#Define some utilities

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example_test(row, path):
    """Encode test images and data to TFRecords"""

    image_filename=str(np.int64(np.floor(row[0])))+'.npz'
    
    filepath=os.path.join(path,image_filename)

    with np.load(filepath) as npz:
      arr = np.ma.MaskedArray(**npz)

    image=arr.data
    mask=arr.mask

    image[mask]=0

    #Recast to more standard channel-last format:
    image=np.transpose(image,[1,2,0])
    height=image.shape[0]
    width=image.shape[1]

    image=image.tobytes()
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image': bytes_feature(image),
        'image/filename': bytes_feature(image_filename.encode('utf8')),
          
    }))
    return tf_example
    
#Encode image and data to TFRecords

def create_tf_example_train(row, path):
    """Encode train images and labels to TFRecords"""
    
    image_filename=str(np.int64(np.floor(row[0])))+'.npz'
    
    filepath=os.path.join(path,image_filename)

    with np.load(filepath) as npz:
      arr = np.ma.MaskedArray(**npz)

    image=arr.data
    mask=arr.mask

    image[mask]=0

    #Recast to more standard channel-last format:
    image=np.transpose(image,[1,2,0])
    height=image.shape[0]
    width=image.shape[1]

    image=image.tobytes()

    P=row[1]
    K=row[2]
    Mg=row[3]
    Ph=row[4]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image': bytes_feature(image),
        'image/filename': bytes_feature(image_filename.encode('utf8')),
        
        'label/P': float_feature(P),
        'label/K': float_feature(K),
        'label/Mg': float_feature(Mg),
        'label/Ph': float_feature(Ph),
          
    }))
    return tf_example