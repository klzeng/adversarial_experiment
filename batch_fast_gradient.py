"""this is a file to generate adversarial sample by using gradient descent"""
from keras.applications import InceptionV3
from keras import backend as K
import numpy as np
import timeit
import sys



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
    start = timeit.default_timer()

    # Your statements here

    
    filepath = r"D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/dataset/"
    img_array_path = filepath + 'image_array.npz'
    img_array = np.load(img_array_path)
    X = img_array['X']
    y_tar = img_array['y_tar']

    


    # build the model and get input and output of the model
    model = InceptionV3()
    model_Input = model.layers[0].input
    model_Output = model.layers[-1].output
    # set the objective type we need to fool the CNN
    X = preprocessing(X)
    hacked_img_array = []
    sample_iter_i = 0
    cost_arr = []
    try:
        for X_t, object_type_to_fake in zip(X, y_tar):
            max_change_above = X_t + 0.01
            max_change_below = X_t - 0.01
    
            # the way to get epsilon is the find the minimal magnitude of the input image
            epsilon = np.min(np.abs(X_t))
    
            # this is to get the code corresponding to the label that we need to fake
            # record that the model output is a 1000 dimentional vector
            cost_to_object = model_Output[0, object_type_to_fake]
            # here we are calculating the derivate of the input since we need to move a little
            # step toward the gradient
            gradient_function = K.gradients(cost_to_object, model_Input)[0]
            grab_cost_and_gradients_from_model = K.function([model_Input, K.learning_phase()],
                                                            [cost_to_object, gradient_function])
    
            #label_t = get_predict_label(X_t, model)
            maxiter = 500
            iter_i = 0
            cost = 0
            # get the movement path from original input to the adversarial input
            noise_move = 0
            while cost < 0.8 and iter_i <= maxiter:
                # get label before update
                cost, gradient_W = grab_cost_and_gradients_from_model([X_t, 0])
                # get the sign of the gradient
                sign_gradient_w = np.sign(gradient_W)
                noise_move += epsilon + sign_gradient_w
                X_t = X_t + epsilon * sign_gradient_w
                X_t = np.clip(X_t, max_change_below, max_change_above)
                X_t = np.clip(X_t, -1.0, 1.0)
                #label_t = get_predict_label(X_t, model)
                iter_i += 1
                #print(iter_i)
            
            cost_arr.append(cost)
            hacked_img_array.append(X_t)
            print(sample_iter_i)
            sample_iter_i += 1
            sys.stdout.flush()
#        hacked_img_array = np.asarray(hacked_img_array)
#        hacked_img_array = recover_from_preprocessing(hacked_img_array)
#        hacked_img_array_file_name = filepath + 'hacked_img_array'
#        np.save(hacked_img_array_file_name, hacked_img_array)
#    
#        stop = timeit.default_timer()
#        
#        cost_arr.append(stop - start)
#        cost_arr_filename = filepath + 'cost_time'
#        np.save(cost_arr_filename, cost_arr)
    except:
        print("code stop at {} iteration".format(sample_iter_i))
    finally:
        hacked_img_array = np.asarray(hacked_img_array)
        hacked_img_array = recover_from_preprocessing(hacked_img_array)
        hacked_img_array_file_name = filepath + 'hacked_img_arraytest1'
        np.save(hacked_img_array_file_name, hacked_img_array)
    
        stop = timeit.default_timer()
        
        cost_arr.append(stop - start)
        cost_arr_filename = filepath + 'cost_time'
        np.save(cost_arr_filename, cost_arr)
        

