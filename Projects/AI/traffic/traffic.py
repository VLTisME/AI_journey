import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    # Hmm, in this code, the author doesn't use batches (segmenting the data into small parts and train them one by one)
    # Instead, he uses the whole data to train the model at once, which is 60% of the data for training and 40% for testing
    # I should try to use batches to see if it is better or not
    labels = tf.keras.utils.to_categorical(labels)


    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    
    """ This is the code to use batches from ChadGPT. Will test it later
    # Convert training data into a TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # Shuffle and batch the training data
    BATCH_SIZE = 32  # Define your batch size
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(BATCH_SIZE)

    # Convert testing data into a TensorFlow dataset and batch it
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    # Optionally, prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    """


    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    dirs_list = os.listdir(data_dir)
    for dir in dirs_list:
        file1 = os.path.join(data_dir, dir)
        dir_list = os.listdir(file1)
        for image in dir_list:
            pic_size = (IMG_WIDTH, IMG_HEIGHT)
            file2 = os.path.join(file1, image)
            pic = cv2.imread(file2)
            res = cv2.resize(pic, pic_size)
            label = int(dir)
            images.append(res)
            labels.append(label)
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
          # Conv la convert? la chuyen doi tu hinh dang nay sang hinh dang khac :))))
          # Nope, actually it is 2D Convolutional Layer
          # The input is 3D but we use 2D Convolutional Layer because 2D Convolutional Layer actually uses some operations applied to 2D data and doesn't care
          # about the third dimension (which is RGB in this case) and simply just applying the filters in EACH CHANNEL separately and the results are combined
          # So when we need to deal with 3D data at once, we use 3D Convolutional Layer, which means an operation considers all 3 dimensions at once
          
          tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)), # Keras recommends to use this line to define the input shape. Do not use "input_shape" in the first layer
          
          tf.keras.layers.Conv2D(8, (5, 5), activation="relu"),
          tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
          tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
          tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
          tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
          tf.keras.layers.Flatten(),
          # Now add first hidden layer
          tf.keras.layers.Dense(128, activation="relu"),
          tf.keras.layers.Dense(128, activation="relu"),
          # remember: each unit "learns" something! (and how it learns? it is kinda like introducing a new function and that function has it own job. TensorFlow will do that)
          tf.keras.layers.Dense(128, activation="relu"),
          tf.keras.layers.Dense(128, activation="relu"),
          tf.keras.layers.Dropout(0.4),
          tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
          optimizer="adam",
          loss="categorical_crossentropy",
          metrics=["accuracy"]
    )

    return model




if __name__ == "__main__":
    main()
