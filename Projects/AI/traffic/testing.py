from tensorflow.keras.models import load_model # type: ignore
import cv2 # type: ignore
import os
import tensorflow as tf # type: ignore
import numpy as np

IMG_WIDTH = 30
IMG_HEIGHT = 30


# Load the saved model
model = load_model("bot.h5")

image_path = os.path.join("gtsrb", "1", "00000_00012.ppm")

# Load the image
image = cv2.imread(image_path)
res = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
# image = np.array(image) / 255.0  # No need to normalize the image since I can see from traffic.py they don't normalize the image
# image = np.array(image)

res = np.array(res)
res = res.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3) # must reshape because the CNN expects a 4D array: (batch_size, height, width, channels), where batch_size is
# The number of images processed in a single forward/backward pass. This can vary, but for inference with one image, it’s set to 1
# height and width: The dimensions of the image (e.g., 30x30 pixels).
# The number of channels: The number of color channels in the image (e.g., 3 for RGB).
# cv2 can't do that because cv2 only resizes the image to a specific width and height, but it doesn't add the batch_size dimension. The output of cv2 is always a 2D or 3D array.
# cv2 works like this: 
# if the image is grayscale, the output is a 2D array (height, width)
# if the image is colored, the output is a 3D array (height, width, channels). For example, because in this problem, the image is colored (RGB), the output is a 3D array. 
# look at line 18: resized_image = cv2.resize(image, (30, 30))  # Output shape: (30, 30, 3)

'''
Why Include batch_size?

Neural networks are optimized for batch processing to increase training and inference efficiency.
Even if you're working with a single image, the model still expects the first dimension to be present, which TensorFlow interprets as the batch size.
'''

prediction = model.predict(res)
# The prediction variable contains the raw output probabilities for each class, represented as a 1D array.
# Shape of prediction: (1, NUM_CATEGORIES), where NUM_CATEGORIES is the total number of classes (e.g., 43 for traffic signs). (1 image, 43 classes) or let's say (1 row 43 columns)
# ex: [[0.1, 0.2, 0.7]] means the model predicts the image belongs to class 2 with a probability of 0.7
# case for batch_size = 2 (2 images)
# prediction = np.array([[0.1, 0.7, 0.2],    # Prediction for the first image
#                       [0.3, 0.4, 0.3]])   # Prediction for the second image
# so it is like 2-dimensional array, where each row represents the prediction for each image. The first row is the prediction for the first image, and the second row is the prediction for the second image.

predicted_class = np.argmax(prediction)
# np.argmax returns the index of the largest value in the array. In this case, the array (prediction) is the output of the model, which is a one-hot encoded array.
# Example for batch_size = 2:
# For the first image
# predicted_class_1 = np.argmax(prediction[0])  # 1 -> "Yield"
# For the second image
# predicted_class_2 = np.argmax(prediction[1])  # 1 -> "Yield"

# print(f"Predicted Class for Image 1: {predicted_class_1}")
# print(f"Predicted Class for Image 2: {predicted_class_2}")

print(f"Predicted Class: {predicted_class}")
