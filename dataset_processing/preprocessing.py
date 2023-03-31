import os, logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

rng1 = tf.random.Generator.from_seed(32)
rng2 = tf.random.Generator.from_seed(10)

def normalize_train_val(patch,label, height, width, max_reflectance, max_labels, mean_labels, std_labels, normalization_mode):
    """Normalize train dataset"""
    patch = tf.cast(patch,tf.float32)
    patch = patch/(max_reflectance)
    
    label = tf.cond(tf.math.equal(normalization_mode, 0), lambda: (label / max_labels), lambda: ((label-mean_labels)/std_labels))

    return patch, label, height, width

def tile_x_train(patch, nx):    
    """Tile patch horizontally with random flips"""
    tiled_patch = tf.image.random_flip_left_right(tf.image.random_flip_up_down(patch, seed=72), seed=64)
    i=0

    def cond(tiled_patch, patch, i, nx):
        return tf.less(i, nx-1)
    def body(tiled_patch, patch, i, nx):
        tiled_patch = tf.concat([tiled_patch, tf.image.random_flip_left_right(tf.image.random_flip_up_down(patch, seed=49), seed=95)], 1)
        i+=1
        return tiled_patch, patch, i, nx

    tiled_patch, _, _, _ = tf.while_loop(cond, body, [tiled_patch, patch, i, nx])

    return tiled_patch
    
def tile_y_train(image,patch, nx, ny):    
    """Tile patch vertically with random flips"""
    i=0

    def cond(image, patch, i, ny):
        return tf.less(i, ny-1)
    def body(image, patch, i, ny):
        image = tf.concat([image, tile_x_train(patch, nx)], 0)
        i+=1
        return image, patch, i, ny
    
    image, _, _, _= tf.while_loop(cond, body, [image, patch, i, ny])
    
    return image

def image_tiling_train(patch, height, width,target_image_size):
    """Tile and flip small patches"""
    nx = tf.math.floordiv(target_image_size, width)
    ny = tf.math.floordiv(target_image_size, height)

    image = tile_y_train(tile_x_train(patch, nx), patch, nx,ny)
    
    image = tf.cond(tf.equal(tf.math.maximum(nx, ny),1), lambda: tf.image.resize(image, [target_image_size, target_image_size], method='bilinear', antialias=False), lambda: tf.image.pad_to_bounding_box(image, 0, 0, target_image_size, target_image_size))

    return image
    
def add_gauss_noise(image,target_image_size,std=0.05):
    """Add gaussian noise"""
    mean = 0

    noise = rng2.normal([target_image_size,target_image_size,150], mean,std)

    noisy_image = image + noise
    
    return noisy_image

def augment_train(patch, label, height, width, target_image_size, noise_std):
    """Augment training images"""
    max_dim = tf.math.maximum(height, width)
    image = tf.cond(max_dim<target_image_size, lambda: image_tiling_train(patch, height, width,target_image_size), lambda: tf.image.resize(patch, [target_image_size, target_image_size], method='bilinear', antialias=False))
    image=add_gauss_noise(image,target_image_size,noise_std)

    index = rng1.uniform(shape=[], maxval=4, dtype=tf.dtypes.int32)
    
    image=tf.cond(tf.equal(index,1),lambda: tf.image.flip_left_right(image), lambda: image)

    image=tf.cond(tf.equal(index,2),lambda: tf.image.flip_up_down(image), lambda: image)

    image=tf.cond(tf.equal(index,3),lambda: tf.image.flip_left_right(tf.image.flip_up_down(image)), lambda: image)

    return image, label

def normalize_test(filename, patch, height, width, max_reflectance):             
    """Normalize test dataset"""
    patch = tf.cast(patch,tf.float32)
    patch = patch/(max_reflectance)

    return filename, patch, height, width

def image_tiling_test(patch, height, width,target_image_size): 
    """Tile small patches"""
    nx = tf.math.floordiv(target_image_size, tf.cast(width, tf.int32))
    ny = tf.math.floordiv(target_image_size, tf.cast(height, tf.int32))
    
    image = tf.tile(patch, [ny, nx, 1])
    image = tf.cond(tf.math.maximum(nx, ny)==1, lambda: tf.image.resize(image, [target_image_size, target_image_size], method='bilinear', antialias=False), lambda: tf.image.pad_to_bounding_box(image, 0, 0, target_image_size, target_image_size))

    return image

def preprocess_test(filename, patch, height, width,target_image_size):
    """Preprocess test set through resizing or tiling"""
    max_dim = tf.math.maximum(height, width)
    image = tf.cond(max_dim<target_image_size, lambda: image_tiling_test(patch, height, width,target_image_size), lambda: tf.image.resize(patch, [target_image_size, target_image_size], method='bilinear', antialias=False))

    return filename, image, height, width
    
def preprocess_val(patch, label, height, width,target_image_size):
    """Preprocess test set through resizing or tiling"""
    max_dim = tf.math.maximum(height, width)
    image = tf.cond(max_dim<target_image_size, lambda: image_tiling_test(patch, height, width,target_image_size), lambda: tf.image.resize(patch, [target_image_size, target_image_size], method='bilinear', antialias=False))

    return image, label