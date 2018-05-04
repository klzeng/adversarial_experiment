
import numpy as np
from keras.applications import resnet50, vgg16, inception_v3
import pandas as pd
from PIL import Image
from keras.preprocessing import image

file_path = r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/dataset/'
data = np.load(r'C:/Users/fzp/Desktop/deepfool_array.npz')

ori_file = pd.read_csv(
    r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/dataset/dev_dataset.csv')

file_names = ori_file['ImageId'].values
ori_vs_after_label = data['ori_vs_after_label']

model = inception_v3.InceptionV3()
img_array = []
for file_names_i in file_names:
    file_names_i = file_path + "adimages/images/" + file_names_i + ".png"
    img = image.load_img(file_names_i, target_size=(299, 299))
    img = image.img_to_array(img)
    img_array.append(img)

img_array = np.array(img_array)
img_array = inception_v3.preprocess_input(img_array)
pre_label = np.argmax(model.predict(img_array), axis=1)

#compare ad target and model predict labels attack success rate
print("target VS model predict: {}".format(np.sum(ori_vs_after_label[:,1] == pre_label)))


#compare original label and model predict label
print("original label VS model predict: ".format(np.sum(ori_vs_after_label[:,0] == pre_label)))

