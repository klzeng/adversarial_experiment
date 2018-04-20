from keras.applications import inception_v3
import keras.backend as K
import numpy as np
import os
import csv
import sys

# root_path =""
root_path = "/scratch/zfeng3/kunling/"

def gen_ad_4_v3(source_dir,dir_num, save_dir, batch_num):
    with open(root_path + "dev_dataset.csv") as csvfile:
        records = {}
        reader = csv.DictReader(csvfile)
        for row in reader:
            toAdd = {"TrueLabel": row["TrueLabel"], "TargetLabel": row["TargetClass"]}
            records[row['ImageId']] = toAdd

    model = inception_v3.InceptionV3()

    # Grab a reference to the first and last layer of the neural net
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    total_image = 0
    success_target = 0
    mis_count = 0

    img_finished = 0
    for img_file in os.listdir(source_dir):
        if not img_file.endswith(".txt"):
            continue

        try:
            img_id = img_file.split(".")[0]
            img = np.loadtxt(source_dir + "/"+img_file)
        except Exception:
            continue

        img = img.reshape(1, 299, 299, 3)
        img = img.astype('float32')

        true_label = int(records[img_id]['TrueLabel'])-1
        target_label = int(records[img_id]['TargetLabel'])-1

        hacked_image = np.copy(img)

        # Pre-calculate the maximum change we will allow to the image
        # We'll make sure our hacked image never goes past this so it doesn't look funny.
        # A larger number produces an image faster but risks more distortion.
        max_change_above = img + 0.01
        max_change_below = img - 0.01

        # How much to update the hacked image in each iteration
        learning_rate = 0.1

        # Define the cost function.
        # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
        cost_function = model_output_layer[0, target_label]

        # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
        # In this case, referring to "model_input_layer" will give us back image we are hacking.
        gradient_function = K.gradients(cost_function, model_input_layer)[0]

        # Create a Keras function that we can call to calculate the current cost and gradient
        grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

        pre_cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

        # In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
        # until it gets to at least 80% confidence
        cost = pre_cost
        count = 0
        total_count = 0
        while cost < 0.80:
            # Check how close the image is to our target class and grab the gradients we
            # can use to push it one more step in that direction.
            # Note: It's really important to pass in '0' for the Keras learning mode here!
            # Keras layers behave differently in prediction vs. train modes!
            cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

            # Move the hacked image one step further towards fooling the model
            hacked_image += gradients * learning_rate

            # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
            hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
            hacked_image = np.clip(hacked_image, -1.0, 1.0)

            count+=1
            total_count+=1
            if total_count%50 == 0:
                print("itre "+ str(total_count) + ": Model's predicted likelihood that the image belongs to class " + str(target_label) + ": {:.8f}%".format(cost * 100))
                sys.stdout.flush()

            if count == 2500 and cost*100 < 0.01:
                learning_rate += 0.3
                count = 0

            if count == 2500 and (cost - pre_cost)*100 < 0.1:
                if learning_rate < 0.5:
                    learning_rate += 0.1
                pre_cost = cost
                count = 0

        print("itre "+ str(total_count) + ": Model's predicted likelihood that the image belongs to class " + str(target_label) + ": {:.8f}%".format(cost * 100))
        img = hacked_image[0]
        img = img.reshape(1, 299,299,3)
        predict_label = np.argmax(model.predict(img))
        if predict_label == target_label:
            success_target +=1
            mis_count += 1
        elif predict_label != true_label:
            mis_count +=1

        total_image += 1

        img = img.reshape(299*299, -1)
        np.savetxt(save_dir + "/" + img_id + "_hacked.txt", img)
        # move hacked image to complete dir
        os.rename(source_dir +"/"+img_file, root_path + "completed/" + img_file)

        # log
        with open(root_path + "log" + dir_num + ".txt", 'w') as logfile:
            to_write = "for source dirctory " + source_dir + ":\n"
            to_write += "total: " + str(total_image) + "\n"
            to_write += "misclassification count: " + str(mis_count) + "\n"
            to_write += "count for getting target label: " + str(success_target) + "\n"
            to_write += "misclassification rate: " + "%.3f"%(float(mis_count)/float(total_image)) + "\n"
            to_write += "get target rate: " + "%.3f"%(float(success_target)/float(total_image)) + "\n\n"
            logfile.write(to_write)

        img_finished += 1
        if img_finished >= int(batch_num):
            break


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "usage: gen_ad_4_v3: dir_num batch_num"
    dir_num = sys.argv[1]
    batch_num = sys.argv[2]

    source_dir = root_path + 'image_arrays' + dir_num
    save_dir = root_path + "ad_4_v3"
    gen_ad_4_v3(source_dir, dir_num, save_dir, batch_num)