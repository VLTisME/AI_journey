1/ Python lists are not suitable for TensorFlow or NumPy operations, as they are not designed for numerical computations or matrix manipulations. -> That's why in your GTSRB code, there is a line to convert from list to np.array but in MNIST, you don't need to :) (because MNIST data is loaded directly using load_data from MNIST, which converts automatically)


4. Why Does the Code Use x_train.shape[0] Instead of a Hard-Coded Number?
Answer: x_train.shape[0] dynamically retrieves the actual number of samples/images in the dataset (not just 1). Hardcoding 1 for the batch size would discard most of the dataset and make it useless for training. For example:

If you have 60,000 training images, x_train.shape[0] will ensure all images are reshaped.
Hardcoding 1 would only reshape and use a single image, which is incorrect for training.
-> Oh lmao in .reshape(1, IMG_WIDTH, IMG_HEIGHT, 3) testing.py in traffic, i used 1 because i test/ train only with 1 image :))


5. Why Does the Code Use x_train.shape[3] Instead of a Fixed Value?
Answer: The x_train.shape[3] ensures the code adapts to the current shape of the dataset. If the dataset already includes a channel dimension (e.g., [number_of_samples, height, width, channels]), this ensures the code doesn’t overwrite or incorrectly assume the number of channels.

-> this raises a question: what is indeed a numpy array? lets google it
What is x_train?
x_train is a collection of images, typically in a numerical array format, where each image is represented by a grid of pixel values. Depending on how the dataset was loaded, x_train could have various shapes.
For example:

For a grayscale dataset (e.g., MNIST):
x_train.shape  # Could be (60000, 28, 28)
60,000 images.
Each image is 28x28 pixels.
No explicit "channel" dimension because grayscale images have only one channel, often left implicit.


For an RGB dataset (e.g., CIFAR-10 or GTSRB):
x_train.shape  # Could be (50000, 32, 32, 3)
50,000 images.
Each image is 32x32 pixels.
The 3 represents the three color channels (Red, Green, Blue).


x_train.shape[0] like above, is the number of images
x_train.shape[1], [2] are width and height
[3] (if exists) is the number of channels




How to access a specific image in a NumPy array:
You can access a specific image in a NumPy array by indexing it like you would with any other list or array. If your x_train array has the shape (num_images, height, width, channels), you can access a specific image using its index.

For example, if you want to access the 5th image in x_train:
Code:
image = x_train[4]  # Indexing starts at 0, so this accesses the 5th image
If x_train.shape == (60000, 28, 28, 1), this will give you an array of shape (28, 28, 1) representing a single grayscale image. The 1 represents the channel dimension, which indicates it’s a grayscale image.

You can access the pixel values of this image (as a 2D grid) by slicing the image:
code:
image_pixels = x_train[4][:, :, 0]  # Access the 2D pixel grid (28x28) of the 5th image
Here:

x_train[4] gives you the entire image (28x28x1).
[:, :, 0] slices out the pixel values, removing the single channel dimension to get a 28x28 grid of pixel values.

An image in a NumPy array is a grid of values:
Yes, an image in a NumPy array is essentially a grid of values where each value corresponds to the brightness of a pixel. In a grayscale image:

Each pixel is represented by a single value, typically ranging from 0 (black) to 255 (white) in 8-bit images. Intermediate values represent different shades of gray.
For example, if an image is 28x28 pixels, the NumPy array representing the image would look like this:

code:
image_pixels = np.array([
    [0, 0, 0, ..., 255],  # First row of pixels
    [128, 255, 0, ..., 128],  # Second row of pixels
    # ... (26 more rows)
])
This is a 2D array (28x28) where each element corresponds to the brightness of one pixel.

my guide:
x_train.shape --> will output something like (number_of_images, image width, image height, channels) --> represent a numpy array of 'number_of_images' images,
each of them is an image size (w, h, channel) (u can treat the channel as the third dimension)
-> to access a cell in a grayscale image of the 5th image, cell row 3, column 5 --> pixel_value = x_train[4][3, 5, 0]  





Why doesn't MaxPooling need an activation function?
MaxPooling does not require an activation function because its purpose is different from the convolution operation. Here's why:

Down-sampling, not transformation:

MaxPooling is not a transformation of the feature map in terms of adding non-linearity or learning. It's simply a way to reduce the size of the feature map by retaining the most important value from each region.
It's essentially a selection operation, where you are not changing the values in any way except by choosing the maximum. The values you keep are already the most "important" ones from that region.
No need for non-linearity:

Since MaxPooling is not performing a weighted sum or any linear operation like a convolution, there's no need to apply a non-linear transformation afterward. MaxPooling inherently focuses on the strongest signals (maximum values), and adding a non-linearity would not benefit the operation or its purpose.
Activation functions (like ReLU, Sigmoid, etc.) are typically used to introduce non-linearity to the network, which is needed for learning complex features. However, MaxPooling is already a non-learnable, fixed operation — it doesn’t need non-linearity to function as intended.
Functionality:

MaxPooling is used to reduce the spatial dimensions (width and height) of the feature map, which reduces the computational load, and helps the model to become more invariant to small translations or distortions in the input. It's essentially a form of dimensionality reduction.
The convolution layer, on the other hand, is responsible for extracting features, and that's why we apply activation functions to introduce non-linearity after the convolution.
Summary:
MaxPooling: A down-sampling operation that reduces the size of the feature map by selecting the maximum value in each window. It does not require an activation function because it’s simply selecting the strongest signal (maximum) and doesn’t involve any transformation of the feature map.
Convolution: A linear operation that needs an activation function (like ReLU) to introduce non-linearity and allow the network to learn complex features.
Thus, while both convolution and pooling are common operations in CNNs, only convolution requires an activation function to allow the model to learn non-linear patterns.


Raised questions:
- How many layers should be added? How many Neurons? What activation function? How many times should we apply maxpooling and convolution?
- How to know if the results are underfitted? overfitted?



# 1 regarding activation function
ReLU (Rectified Linear Unit):
Default for hidden layers. Works well in most cases due to its simplicity and computational efficiency.
Helps with vanishing gradient issues in deep networks.
Sigmoid/Tanh:
Useful for the output layer in specific cases (e.g., sigmoid for binary classification, tanh for regression with normalized outputs).
Avoid for hidden layers in deep networks (they can cause vanishing gradients).
Softmax:
Use for the output layer in multi-class classification.
Leaky ReLU or ELU:
Variants of ReLU to mitigate "dying ReLU" problems.

# 2 regarding neurons
u could try the later layer, the more neurons: 32 - 64 - 128...


# 3 regarding how many layers
the simpler the task/ dataset -> the less layer and vice versal


# 4 regarding how to recognize that is is underfitting or overfitting
Underfit:
the model is too simple with too complex data/ problem and vice versal
Overfit:
the model performs poorly on new dataset, data that it has never seen