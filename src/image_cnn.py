from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil

class ModelImageCNN():
    def __init__(self, verbose=True):
        self.verbose = verbose

    def train(self, train_t, train_n, val_t, val_n, batch_size, epochs, path=None):
        # model definition
        model = Sequential()
        model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(80, 80, 1)))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(filters=12, kernel_size=(5, 5), activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(units=120, activation = 'relu'))
        model.add(layers.Dense(units=80, activation = 'relu'))
        model.add(layers.Dense(units=1, activation = 'sigmoid'))

        if self.verbose:
            model.summary()

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        # prepare train data
        x_train = np.r_[train_t, train_n]
        y_train = np.r_[np.ones(len(train_t)), np.zeros(len(train_n))]

        x_train, y_train = shuffle(x_train, y_train)

        # prepare validation data
        x_test = np.r_[val_t, val_n]
        y_test = np.r_[np.ones(len(val_t)), np.zeros(len(val_n))]

        x_test, y_test = shuffle(x_test, y_test)

        x_train = np.r_[x_train].reshape(x_train.shape[0], 80, 80, 1)
        y_train = np.r_[y_train].reshape(y_train.shape[0], 1)
        x_test = np.r_[x_test].reshape(x_test.shape[0], 80, 80, 1)
        y_test = np.r_[y_test].reshape(y_test.shape[0], 1)

        history = []

        # train model on BALANCED data
        if path:
            # safe model with lowest validation loss
            checkpoint = tf.keras.callbacks.ModelCheckpoint( path,
                                                             monitor='val_loss',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             save_weights_only=False,
                                                             mode='auto',
                                                             save_frequency=1)

            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(x_test, y_test),
                                callbacks=[checkpoint])
        else:
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(x_test, y_test))

        # print learning graph
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        plt.plot(range(1, len(acc)+1), acc, 'b', label='Training accuracy')
        plt.plot(range(1, len(acc)+1), val_acc, 'r', label='Validation accuracy')
        plt.title('CNN training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        self.model = model

    def test(self, test):
        result = {}

        for file, image in test.items():
            file = ''.join(file.split('/')[-1].split('.')[:-1])
            image = image.reshape(1,80,80)

            # make predictions
            score = self.model.predict([image])[0][0]
            result[file] = (score, score > 0.5)

        return result

    def save(self, path):
        # remove the file if it already exists
        if os.path.exists(path):
            shutil.rmtree(path)

        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
