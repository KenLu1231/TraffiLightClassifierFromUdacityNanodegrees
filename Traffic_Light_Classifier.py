import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[0][0]
selected_label = IMAGE_LIST[0][1]

for im in IMAGE_LIST:
    if im[1] == "yellow":
        selected_image = im[0]
        selected_label = im[1]
        break

print("Shape: " + str(selected_image.shape))
print("Label : " + str(selected_label))
plt.imshow(selected_image)

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    
    return standard_im

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [0, 0, 0]
    
    if label == "red":
        one_hot_encoded[0] = 1
    elif label == "yellow":
        one_hot_encoded[1] = 1
    else:
        one_hot_encoded[2] = 1
    
    return one_hot_encoded

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

standard_image = STANDARDIZED_LIST[0][0]
standard_label = STANDARDIZED_LIST[0][1]

print("Shape: " + str(standard_image.shape))
print("Label : " + str(standard_label))
plt.imshow(standard_image)

# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

def create_feature(rgb_image):
    
    img_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    img_equa = cv2.equalizeHist(img_gray)
    img_rgbe = cv2.cvtColor(img_equa, cv2.COLOR_GRAY2RGB)
    
    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(img_rgbe, cv2.COLOR_RGB2HSV)
    
    croped_img = crop_image(hsv)
    
    masked_img = mask_image(croped_img)
    
    ## TODO: Create and return a feature value and/or vector
    part_root = 3
    part_nums = int(len(masked_img) / part_root)
    feature = []
    
    for i in range(part_root):
        t_value = 0
        for parts in masked_img[(i * part_nums):((i + 1)* part_nums)]:
            for row in parts:
                for value in row:
                    t_value += value
        feature.append(t_value)
    
    return feature

def mask_image(image):
    ## Mask image
    lower_bound = np.array([0, 10, 20]) 
    upper_bound = np.array([255, 120, 215])
    
    mask = cv2.inRange(image, lower_bound, upper_bound)
    image[mask != 0] = [0, 0, 0]
    
    return image
    
def crop_image(image):
    ## Crop image
    col_crop = 4
    row_crop = 14
    image = image[col_crop:-col_crop, row_crop:-row_crop, :]
    
    return image

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    feature = create_feature(rgb_image)
        
    predicted_label = [0, 0, 0]
    predicted_label[feature.index(max(feature))] = 1
    
    return predicted_label

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Visualize misclassified example(s)
index = 0
f, miss_list = plt.subplots(1, len(MISCLASSIFIED), figsize=(20,10))
for ax in miss_list:
    miss_label = MISCLASSIFIED[index][1]
    ax.set_title(miss_label)
    ax.imshow(MISCLASSIFIED[index][0])
    index = index + 1


# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")

