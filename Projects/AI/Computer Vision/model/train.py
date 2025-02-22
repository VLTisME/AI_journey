from network import *

history = model.fit(
    train_dataset,
    validation_data = test_dataset,
    epochs = 165,
    callbacks = callbacks
)


model.summary()
result = model.evaluate(test_dataset, verbose = 2)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4))

ax[0].plot(history.history['accuracy'], label = 'accuracy')
ax[0].plot(history.history['val_accuracy'], label = 'val_accuracy')
ax[0].set(xlabel = 'epoch', ylabel = 'accuracy', title = 'Training and Validation accuracy')
ax[0].legend(loc = 'lower right')

ax[1].plot(history.history['loss'], label = 'loss')
ax[1].plot(history.history['val_loss'], label = 'val_loss')
ax[1].set(xlabel = 'epoch', ylabel = 'loss', title = 'Training and Validation loss')
ax[1].legend(loc = 'lower right')