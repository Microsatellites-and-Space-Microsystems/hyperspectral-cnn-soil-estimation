import os, time, cv2, sys
import numpy as np
import tflite_runtime.interpreter as tf_lite
import pandas as pd

sys.path.append('/home/usr/local/lib/python3.7/dist-packages')

# Define directories
dataset_dir = '/home/user/sd/inference/dataset/'
model_dir = '/home/user/sd/inference/TFlite_networks/'

# Preprocessing values
max_reflectance_overall = 6315
max_gt_values = [325, 625, 400, 14]
target_image_size = 32

# Create lists to load all test images and models
npz_images = []
models = []

for filename in os.listdir(dataset_dir):
    num_str = filename.split('.')[0]
    if num_str.isdigit():
        num = int(num_str)
        fullpath = os.path.join(dataset_dir,filename)
        npz_images.append((num, fullpath))

npz_images.sort()
npz_images = [filepath for num, filepath in npz_images]

for model_path in os.listdir(model_dir):
    full_path = os.path.join(model_dir,model_path)
    if os.path.isfile(full_path):
        models.append(full_path)

######################################################################
# Preprocessing functions
# The preprocessing functions have been adapted so that TensorFlow is not required on the Coral Dev Board Mini
def normalize(image, max_reflectance_overall):
    
    image = np.divide(image,max_reflectance_overall,dtype=np.float32)

    return image

def image_repetition_test(image, height, width):    
    nx = np.floor_divide(target_image_size, width)
    ny = np.floor_divide(target_image_size, height)
    
    image = np.tile(image, [ny, nx, 1])

    if np.maximum(nx, ny)==1:
        image = cv2.resize(image, [target_image_size, target_image_size], interpolation=cv2.INTER_LINEAR)
    else:
        [height, width] = np.shape(image)[:2]
        image = np.pad(image, ((0,target_image_size-height),(0,target_image_size-width),(0,0)), 'constant', constant_values=0)
    return image
    
def pad_with_patches_test(image, height, width):
    max_dim = np.maximum(height, width)
    
    if max_dim < target_image_size:
        image = image_repetition_test(image, height, width)
    else:
        image = cv2.resize(image, [target_image_size, target_image_size], interpolation=cv2.INTER_LINEAR)

    return image
######################################################################

for model_path in models:

    # Initialize variables to export data
    num_images = len(npz_images)
    predictions = np.zeros((num_images, 4))
    
    image_loading_time = np.zeros((num_images,1))
    image_overall_time = np.zeros((num_images,1))
    image_preprocessing_time = np.zeros((num_images,1))
    image_processing_time = np.zeros((num_images,1))
    network_inference_time = np.zeros((num_images,1))
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    network = tf_lite.Interpreter(model_path, num_threads=4)
    network.allocate_tensors()
    network.invoke()
    network_input_details = network.get_input_details()[0]
    network_output_details = network.get_output_details()[0]

    print('Starting inference of model: ' + model_name)
    i=-1
    for image_path in npz_images:
        i += 1

        # Loading
        image_time_start = time.time()  # Time start

        with np.load(image_path) as npz:
            arr = np.ma.MaskedArray(**npz)

        image_loading_time[i,0] = time.time() - image_time_start   # Loading time

        image = arr.data
        mask = arr.mask
        image[mask] = 0

        image = np.transpose(image,[1,2,0])
        h = np.shape(image)[0]
        w = np.shape(image)[1]

        image_processing_time_start = time.time()   # Processing time start

        # Preprocessing
        image = normalize(image, max_reflectance_overall)
        image = pad_with_patches_test(image, h, w)

        image = np.expand_dims(image,0).astype(network_input_details["dtype"])

        image_preprocessing_time[i,0] = time.time() - image_processing_time_start

        # Predictions
        inferece_time_start = time.time()   # Inference time start

        network.set_tensor(network_input_details['index'], image)
        network.invoke()
        pred = network.get_tensor(network_output_details['index'])[0]

        network_inference_time[i,0] = time.time() - inferece_time_start    # Inference time

        predictions[i,:] = pred    
        image_processing_time[i,0] = time.time() - image_processing_time_start   # Processing time
        image_overall_time[i,0] = time.time() - image_time_start   # Overall time
    

    base_path = '/home/user/sd/inference/coral_inference/results/'
        
    submission_dir = base_path + 'results_' + model_name
    if not os.path.exists(submission_dir):
        os.mkdir(submission_dir)
    
    # Export submission file
    predictions *= max_gt_values
        
    submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
    submission.to_csv(os.path.join(submission_dir, 'submission_' + model_name + '.csv'), index_label="sample_index")
    
    # Export time data
    keys = ['image loading time','image preprocessing time','network inference time','image processing time','image overall time']
    time_log = np.concatenate((image_loading_time, image_preprocessing_time, network_inference_time, image_processing_time, image_overall_time), axis = 1)
    
    time_file = pd.DataFrame(time_log, columns=keys)
    time_file.to_csv(os.path.join(submission_dir, 'times_' + model_name + '.csv'), index_label='Image index')

    total_times = np.sum(time_log, axis = 0)

    time_stats = {
        'total loading time': total_times[0],
        'total preprocessing time': total_times[1],
        'total inference time': total_times[2],
        'total processing time': total_times[3],
        'total time': total_times[4],
        'fps network inference': num_images/total_times[2],
        'fps processing': num_images/total_times[3],
        'fps overall': num_images/total_times[4]
        }
    time_stats = pd.DataFrame.from_dict(time_stats,orient='index') 
    time_stats.to_csv(os.path.join(submission_dir, 'time_stats_' + model_name + '.txt'), header=False)

    print('Successfully completed inference of model: ' + model_name)
    print('-----------------------------')