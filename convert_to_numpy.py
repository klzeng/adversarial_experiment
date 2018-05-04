"""This is the code to convert the input image into numpy array"""

import numpy as np
import os
import pandas as pd
from keras.preprocessing import image


def main(fpath):
    image_path = os.path.join(fpath, 'images')
    image_info_path = fpath + 'dev_dataset.csv'
    image_info_pd = pd.read_csv(image_info_path)
    y_tru_laebls = image_info_pd.TrueLabel.values
    y_target_labels = image_info_pd.TargetClass.values
    file_names = image_info_pd.ImageId.values
    X = []
    for each_file in file_names:
        each_img_path = os.path.join(image_path, each_file) + r'.png'
        with image.load_img(each_img_path, target_size=(299, 299)) as img:
            img = image.img_to_array(img)
            X.append(img)
    X = np.asarray(X)
    array_output_path = fpath + 'image_array'
    np.savez(array_output_path, X=X, y_tru=y_tru_laebls, y_tar=y_target_labels)







if __name__ == "__main__":
    filepath = r"D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/dataset/"
    main(filepath)