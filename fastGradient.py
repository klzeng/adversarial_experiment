from keras.applications import InceptionV3
from keras.preprocessing import image
from keras import backend as K
from PIL import Image

import numpy as np
import tensorflow as tf


# before feed in the inceptionV3 model we need to rescale the input ---> [-1, 1]
def preprocessing(data):
    data /= 255.
    data -= 0.5
    data *= 2.0
    return data


def recover_from_preprocessing(data):
    data /= 2.0
    data += 0.5
    data *= 255.
    return data


# return the predict label
def get_predict_label(X, model):
    return np.argmax(model.predict(X), axis=1)

if __name__ == "__main__":
    # load the image
    img = image.load_img("C:\\Users\\fzp\\Desktop/1.jpg",
                         target_size=(229, 229))

    original_image = image.img_to_array(img)
    original_image = preprocessing(original_image)
    original_image = np.expand_dims(original_image, axis=0)
    # build the model and get input and output of the model
    model = InceptionV3()
    model_Input = model.layers[0].input
    model_Output = model.layers[-1].output
    # set the objective type we need to fool the CNN
    object_type_to_fake = 859
    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01
    # copy the original image
    hacked_image = np.copy(original_image)

    # the way to get epsilon is the find the minimal magnitude of the input image
    epsilon = np.min(np.abs(hacked_image))

    # this is to get the code corresponding to the label that we need to fake
    # record that the model output is a 1000 dimentional vector
    cost_to_object = model_Output[0, object_type_to_fake]
    # here we are calculating the derivate of the input since we need to move a little
    # step toward the gradient
    gradient_function = K.gradients(cost_to_object, model_Input)[0]
    grab_cost_and_gradients_from_model = K.function([model_Input, K.learning_phase()],
                                                    [cost_to_object, gradient_function])

    X_t = hacked_image.copy()
    label_t = get_predict_label(X_t, model)
    maxiter = 10000
    i = 0
    cost = 0
    # get the movement path from original input to the adversarial input
    noise_move = 0
    while cost < 0.8 and i <= maxiter:
        # get label before update
        cost, gradient_W = grab_cost_and_gradients_from_model([X_t, 0])
        # get the sign of the gradient
        sign_gradient_w = np.sign(gradient_W)
        noise_move += epsilon + sign_gradient_w
        X_t = X_t + epsilon * sign_gradient_w
        X_t = np.clip(X_t, max_change_below, max_change_above)
        X_t = np.clip(X_t, -1.0, 1.0)
        label_t = get_predict_label(X_t, model)

        i += 1
        print('cost:{0:.5f},    number of iteration:{1}'.format(cost, i))

    X_t = recover_from_preprocessing(X_t)
    img = X_t[0]
    im = Image.fromarray(img.astype(np.uint8))



