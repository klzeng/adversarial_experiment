import numpy as np
from keras.applications import inception_v3, vgg16
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
from PIL import Image

file_path = "D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/hacked_img_array/"
X = np.zeros([0, 1, 299, 299, 3])
for i in range(0, 1000, 20):
    try:
        mini_X = np.load(file_path + "hacked_img_array" + str(i + 20) + ".npy")
        X = np.append(X, mini_X, axis=0)
    except:
        print(i + 20)
        continue

data = np.load(
    r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/dataset/image_array.npz')
y_tru = data['y_tru']
y_tar = data['y_tar']

y_tru = y_tru - 1
y_tar = y_tar - 1

y_tar_shring = np.append(y_tar[:860], y_tar[880:])
y_tru_shring = np.append(y_tru[:860], y_tru[880:])

y_tar_shring = np.append(y_tar_shring[:280], y_tar_shring[380:])
y_tru_shring = np.append(y_tru_shring[:280], y_tru_shring[380:])

X = np.squeeze(X, axis=1)
for j, i in enumerate(X):
    im = Image.fromarray(i.astype(np.uint8))
    im.save(
        r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/InceptionV3_FG/' + str(j) + '.jpg')

X_ad = []
for i in range(880):
    img = image.load_img(
        r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/InceptionV3_FG/' + str(i) + '.jpg',
        target_size=(224, 224))
    X_ad.append(image.img_to_array(img))

X_ad = np.array(X_ad)

X_ad = vgg16.preprocess_input(X_ad)
model = vgg16.VGG16()

pre_label = np.argmax(model.predict(X_ad), axis=1)

np.sum(pre_label == y_tar_shring)
np.sum(pre_label == y_tru_shring)
