import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models, datasets, Model, Layer # type: ignore
import tensorflow_datasets as tfds # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore


import matplotlib.pyplot as plt # type: ignore
import numpy as np



def lr_schedule(epoch):
    iterations_per_epoch = 50000 // 128  
    if epoch < 82:  
        return 0.1  
    elif epoch < 123:  
        return 0.01  
    else:  
        return 0.001  

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose = 1)

optimizer = tf.keras.optimizers.SGD(
    learning_rate = 0.1,
    weight_decay = 1e-4,    
    momentum = 0.9
)

# early_stopping = EarlyStopping(  
#     monitor='val_loss',         # Watch validation loss  
#     patience=10,                # Wait for 10 epochs before stopping  
#     restore_best_weights=True   # Restore the best weights at the end  
# )  

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(  
#     monitor = "val_loss",        # Monitor validation loss  
#     factor = 0.1,                # Reduce learning rate by this factor (e.g., divide by 10)  
#     patience = 5,                # Number of epochs with no improvement before reducing LR  
#     verbose = 1,                 # Print LR reduction messages  
#     mode = "min",                # Aim to minimize the monitored metric  
#     min_lr = 1e-5                # Lower bound for learning rate  
# )  

checkpoint = ModelCheckpoint(  
    filepath = 'best_model.keras', 
    monitor = 'val_accuracy',      # Save based on validation accuracy  
    save_best_only=True,         # Save the model only if it improves  
    mode = 'max'                   # We want max accuracy  
)

# Combine everything into a list of callbacks  
callbacks = [lr_scheduler, checkpoint]

model.compile( # type: ignore
    optimizer = optimizer,
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), # didn't 1 hot encode so sparse, didn't softmax so from logits
    metrics = ['accuracy']
)



