from keras.models import load_model
import numpy as np
from keras.datasets import mnist
from keras import models
from keras import backend as K


def preprocessing(data):
    data /= 255.
#    data -= 0.5
#    data *= 2.0
    return data


def recover_from_preprocessing(data):
#    data /= 2.0
#    data += 0.5
    data *= 255.
    return data

def get_predict_label(X, model):
    return np.argmax(model.predict(X), axis=1)


if __name__ == "__main__":
    model = load_model(r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/gittry/adversarial_experiment/mnist_model1')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28,28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28,28,1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28,28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #x_train = preprocessing(x_train)
    #x_test = preprocessing(x_test)

    # move every true label one step right
    object_fake = np.copy(y_train) + 1
    object_fake[object_fake == 10] = 0

    number_sample = 500
    input_imgs = np.copy(x_train[:number_sample])
    object_fake_sample = np.copy(object_fake[:number_sample])

    model_Input = model.layers[0].input
    model_output = model.layers[-1].output
    maxiter = 10000
    sample_i = 0
    input_imgs = np.expand_dims(input_imgs, axis=1)
    object_fake_sample += 1
    object_fake_sample[object_fake_sample==10] = 0
    lam = 0.05
    eta = 1.0
    for hacked_img, obj_lable in zip(input_imgs, object_fake_sample):
        #epsilon = np.min(np.abs(hacked_img))
        X_return = np.random.normal(np.mean(hacked_img), np.std(hacked_img), hacked_img.shape)
        cost_to_object = model_output[0, obj_lable]
        gradient_function = K.gradients(cost_to_object, model_Input)[0]
        grab_cost_andgradients_from_model = K.function([model_Input, K.learning_phase()],
                                                       [cost_to_object, gradient_function])

        i = 0
        cost = 0
        # max_change_above = hacked_img + 0.01
        # max_change_below = hacked_img - 0.01
        while cost < 0.8 and i <= maxiter:
            cost, gradient_W = grab_cost_andgradients_from_model([X_return, 0])
            #sign_gradient_w = np.sign(gradient_W)
            X_return += eta * (gradient_W + lam * (X_return - hacked_img))
            
            
            i += 1
            if i % 500 == 0:
                print('cost:{0:.10f},    number of iteration:{1}'.format(cost, i))

        print('sample {}: '.format(sample_i))
    




