from keras.preprocessing import image
import os
import numpy as np
import sys

# this will generate a txt file storing your the array
# presentation of your image in shape (-1, 3) for each image
# in src_dir.
# all generated txt file will store in target dir

def save_img_array_2_file(src_dir, target_dir, target_size):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    dim1 = int(target_size[0])
    dim2 = int(target_size[1])
    for img_name in os.listdir(src_dir):
        try:
            img = image.load_img(src_dir + "/"+ img_name, target_size=(dim1, dim2))
            img_id = img_name.split(".")[0]
            img = image.img_to_array(img)
            #
            # do your preprocessing here if you want
            #
            img = img.reshape(dim1*dim2, -1)
            np.savetxt(target_dir + "/" + img_id+".txt", img)
            print "saved array of image " + img_name
        except IOError:
            continue

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "usage: python save_2_file source_dir target_dir target_size(dim1, dim2, dim3)"
        sys.exit(0)

    src_dir = sys.argv[1]
    target_dir = sys.argv[2]
    target_size = sys.argv[3][1:-1].split(",")

    save_img_array_2_file(src_dir,target_dir, target_size)