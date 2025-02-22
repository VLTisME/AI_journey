from settings import *

# ----------------------- Load data --------------------------------------

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(f"Training dataset has shape: {train_images.shape}, {train_labels.shape}")
print(f"Testing dataset has shape: {test_images.shape}, {test_labels.shape}")

classes = np.unique(train_labels)
print(classes)
class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize = (10, 10))
for i in range(10):
    for j in range(train_images.shape[0]):
        if train_labels[j, 0] == i:
            plt.subplot(2, 5, i + 1)
            plt.imshow(train_images[j])
            plt.title(class_name[i])
            plt.axis('off')
            break
        
        
# -------------------------- Data Augmentation -------------------------------------
mean = tf.constant([0.4914, 0.4822, 0.4465])  
std = tf.constant([0.2023, 0.1994, 0.2010])  

def normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    # broadcasting magic here: (32, 32, 3) with (3, ) -> automatically broadcasting to (32, 32, 3) with (32, 32, 3)
    return (image - mean) / std

def cutout(image, mask_size = 8):
    h, w = tf.shape(image)[0], tf.shape(image)[1]

    cutout_center_x = tf.random.uniform(shape = [], minval = 0, maxval = h, dtype = tf.int32)
    cutout_center_y = tf.random.uniform(shape = [], minval = 0, maxval = w, dtype = tf.int32)

    lower_x = tf.maximum(0, cutout_center_x - mask_size // 2)
    upper_x = tf.minimum(h, cutout_center_x + mask_size // 2)
    lower_y = tf.maximum(0, cutout_center_y - mask_size // 2)
    upper_y = tf.minimum(w, cutout_center_y + mask_size // 2)

    mask = tf.ones((h, w), dtype=tf.float32)  
    # doan nay hoi kho hieu
    mask = tf.tensor_scatter_nd_update(  
        mask,
        indices = tf.reshape(tf.stack(tf.meshgrid(tf.range(lower_x, upper_x), tf.range(lower_y, upper_y), indexing='ij'), axis=-1), [-1, 2]),  
        # tf.meshgrid(..., ...) thi no tao ra 2 thang [ [] ], [ [] ] truoc, xong stack no se them 1 chieu, va chieu do la axis = -1 (chieu cuoi cung) 2 thang do vao chieu thu 3
        # reshape[-1, 2] thi cung nhu (2, 3, 2) xong (-1, 2) -> (6, 2)
        updates = tf.zeros((upper_x - lower_x) * (upper_y - lower_y))  # create a vector of zeros
        # mask = tf.tensor_scatter_nd_update(mask, indices, updates)  
    )  

    # tf doesnt automatically broadcasting lol
    mask = tf.expand_dims(mask, axis = -1) # now (32, 32, 1)
    mask = tf.broadcast_to(mask, tf.shape(image))
    cutout_image = tf.cast(image, tf.float32) * mask

    return tf.cast(cutout_image, tf.uint8)

def random_rotation(image):
    image = tf.image.rot90(image, k = tf.cast(tf.random.uniform([], 0, 4), tf.int32))
    return image

def preprocess_train(image, label):
    label = tf.squeeze(label) # squeeze basically just converts all skeleton dim to disappear
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)
    image = tf.image.random_crop(image, size = [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    # image = cutout(image)
    # image = random_rotation(image)
    # image = tf.image.random_brightness(image, max_delta = 0.1)
    # image = tf.image.random_contrast(image, lower = 0.8, upper = 1.2)
    # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  
    # image = tf.image.random_hue(image, max_delta=0.1)
    # image = tf.image.central_crop(image, central_fraction=0.8)  # Crop the central portion  
    # image = tf.image.resize(image, [32, 32])  # Resize back to 32x32  
    
    return normalize_image(image), label

def preprocess_test(image, label):
    label = tf.squeeze(label) # transform from (128, 1) to (128, )
    return normalize_image(image), label

def visualize_augmentations(dataset, num_images = 10):
    plt.figure(figsize = (15, 5))
    dataset = dataset.unbatch() # unbatch because it was batched into multiple (128, 32, 32, 3)s -> each element is a (128, 32, 32, 3)
    for i, (augmented_image, label) in enumerate(dataset.take(num_images)):
        plt.subplot(1, num_images, i + 1)
        #plt.imshow((augmented_image.numpy() * std + mean).clip(0, 1))
        plt.imshow(augmented_image)
        #print(augmented_image.shape)
        #print(label.shape)
        plt.xticks([])
        plt.yticks([])
        plt.title(class_name[label])

batch_size = 128


# the resulting train_dataset still has 50000 images
# the magic of data augmentation here is for each epoch, it will "random" augment data so the model
# sees the data differently in each epoch!!! so magical
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .map(preprocess_train, num_parallel_calls = tf.data.AUTOTUNE)
    .shuffle(buffer_size = 50000) # each epoch will load the dataset pipeline again so the data is shuffled again every epoch
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
# Why prefer mini-batches? because we utilize GPU or TPU's parallelism to handle multiple batches at once!
# that's why batch(batch_size) is used in both train_dataset and test_dataset
# also mini batches SGD brings faster convergence for training process
test_dataset = (
    tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    .map(preprocess_test, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

visualize_augmentations(train_dataset, 10)
for (image, label) in train_dataset.take(1):
    print(image.shape)
    print(label.shape)