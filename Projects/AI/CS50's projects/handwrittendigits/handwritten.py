import sys
import tensorflow as tf  # type: ignore

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2], 1)),

        tf.keras.layers.Conv2D(8, (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation = "relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(10, activation = "softmax")
])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor the validation accuracy - la no xem xet thang accuracy tren tap validation
    min_delta=0.05,           # Minimum change to qualify as an improvement
    patience=2,              # Stop after 2 epochs with no improvement
    restore_best_weights=True  # Restore the model with the best validation accuracy
)

model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]  
)

model.fit(x_train, y_train, 
          epochs = 10, 
          # validation_data = (x_test, y_test), 
          # callbacks = [early_stopping], 
          # verbose = 2
        )
model.evaluate(x_test, y_test, verbose = 2)

if (len(sys.argv) == 2):
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}!")

