import numpy as np
import keras
from keras import models, backend
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv3D, MaxPool2D


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28,28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28,28,1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28,28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


    model = models.Sequential()
    model.add(Conv2D(20,kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(40, (5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer = 'Adam',
                  metrics = ['accuracy'])

    model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=1, validation_split=0.17)
    model.save(r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/gittry/adversarial_experiment/mnist_model1')
    score = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
    print ('Test loss: ', score[0])
    print ('Test accuracy: ', score[1])
