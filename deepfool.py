import numpy as np
from keras.applications import InceptionV3
import sys
from keras import backend as K
#from PIL import Image
import tensorflow as tf


# return the predict label
def get_predict_label(X, model):
    return np.argmax(model.predict(X), axis=1)

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


if __name__ == "__main__":

    number_class = 10
    file_path = r'/scratch/zfeng3/zhanpeng/dfalgo/'
    model = InceptionV3()
    model_input = model.layers[0].input
    model_output = model.layers[-1].output
    w = np.zeros([299, 299, 3])
    
    x_whole = np.load(r'/scratch/zfeng3/zhanpeng/image_array.npz')
    #x_whole = np.load(r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/dataset/image_array.npz')
    
    xs = x_whole['X']
   
    xs = preprocessing(xs)
    deepfool_list = []
    r_tot_list = []
    ori_vs_ad = []
    lr = 0.02
    for x in xs:
        x_ori = np.copy(x)
        iter_i = 0
        max_iter = 1000
        r_change = []
        f_k_list = model.predict(x)[0].argsort()[::-1]
        x_ori_label = f_k_list[0]
        rest_labels = f_k_list[1:number_class]
        x_label = x_ori_label
        r_tot = np.zeros([1, 299, 299, 3])
        while x_ori_label == x_label and iter_i < max_iter:
            pert_min = np.inf
            cost_function = model_output[0, x_ori_label]
            gradient_function = K.gradients(cost_function, model_input)[0]
            grab_cost_and_gradients_from_model = K.function([model_input, K.learning_phase()], [gradient_function])
            original_grad = grab_cost_and_gradients_from_model([x_ori, 0])[0]
            for k in rest_labels:
                cost_function = model_output[0, k]
                gradient_function = K.gradients(cost_function, model_input)[0]
                grab_cost_and_gradients_from_model = K.function([model_input, K.learning_phase()], [gradient_function])
                curr_grad = grab_cost_and_gradients_from_model([x, 0])[0]
                w_k_prime = curr_grad - original_grad
                f_k_curr = abs(f_k_list[k] - f_k_list[x_ori_label])
                
                
                pert_k = f_k_curr / np.linalg.norm(w_k_prime.flatten())
                if pert_k < pert_min:
                    pert_min = pert_k
                    w = w_k_prime
                print('k is: {}'.format(k))
                sys.stdout.flush()
            r_i = (pert_min + 1e-4) * w / np.linalg.norm(w.flatten())
            print('sum r_i: {}'.format(np.sum(r_i)))
            sys.stdout.flush()    
           
            r_tot += r_i
            x = x_ori + (1 + lr) * r_tot
            f_k_list = model.predict(x)[0]
            x_label = np.argmax(f_k_list)
            print('iter is : {}'.format(iter_i))
            iter_i += 1
        r_tot += (1+lr) * r_tot
        deepfool_list.append(x)
        r_tot_list.append(r_tot)
        ori_vs_ad.append((x_ori_label, x_label))
        if iter_i % 50 == 0:
            deepfool_list = np.array(deepfool_list)
            deepfool_list = recover_from_preprocessing(deepfool_list)
            deepfool_list = np.clip(deepfool_list, 0, 255)
            r_tot_list = np.array(r_tot_list)
            ori_vs_ad = np.array(ori_vs_ad)
            file_path_iter = file_path + "dfalgo" + str(iter_i)
            np.savez(x = deepfool_list, rot = r_tot_list, ori_vs_ad = ori_vs_ad, file_path_iter)
            deepfool_list = []
            r_tot_list = []
            ori_vs_ad = []
            
    #save as a np big array x=ad sample, rot=permtuation range ori_vs_ad is the label or the original image and the ad images
        #np.save('/scratch/zfeng3/zhanpeng/deepfool_img', x)
        

