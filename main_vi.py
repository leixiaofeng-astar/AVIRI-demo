import sys
import time

from torchvision import transforms, datasets
from torch import cuda
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn import metrics
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import glob
import re
import matplotlib.pyplot as plt
import json
import pdb
import cv2
import torch.nn as nn
import argparse
import pydicom
from imgaug import augmenters as iaa
import imgaug as ia
import torch.nn.functional as F
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import h5py as h5
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras import backend as K


#prj_path = '/Users/test/sunny/nCoV-detection/ModStore/'
prj_path = './'
modeldir = prj_path + 'test_model/'
testdir = prj_path + 'images/'
outputsdir = prj_path + 'outputs/'
cp_filepath = prj_path + 'best_model.h5'

error_code = 0

def nparray2tensor(x, cuda=True):
    normalize = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if x.shape[-1] == 1:
        x = np.tile(x, [1, 1, 3])

    # ToTensor() expects [h, w, c], so transpose first
    x2 = normalize(transforms.ToPILImage()(x.copy()))
    return x2


def is_img(ext):
    ext = ext.lower()
    if ext == '.jpg' or ext == '.JPG' :
        return True
    elif ext == '.png' or ext == '.PNG':
        return True
    elif ext == '.jpeg' or ext == '.JPEG':
        return True
    elif ext == '.bmp' or ext == '.BMP':
        return True
    elif ext == '.tif' or ext == '.TIF':
        return True
    elif ext == '.tiff' or ext == '.TIFF':
        return True
    elif ext == '.dcm' or ext == '.dicom':
        return True
    elif ext == '.DCM' or ext == '.DICOM':
        return True
    else:
        return False

def readImageFile(FilewithPath):
    # we could repeat a few times in case the network file transder is not done
    img = None
    fileName = os.path.basename(FilewithPath)
    ext_str = os.path.splitext(fileName)[1]
    repeat_time = 3
    if is_img(ext_str):
        print("Input Image: ", fileName)
        cycle_cnt = repeat_time
        while cycle_cnt>0 and img is None:
            # try one more time in case libpng error: Read Error
            try:
                if ext_str == '.dcm' or ext_str == '.dicom' or ext_str == '.DCM' or ext_str == '.DICOM':
                    ds = pydicom.read_file(FilewithPath)  # read dicom image
                    img = ds.pixel_array  # get image array
                    img = np.array(img)
                    '''
                    When using pixel_array with Pixel Data that has an (0028,0002) Samples per Pixel value of 3 
                    then the returned pixel data will be in the color space as given by (0028,0004) Photometric Interpretation 
                    (e.g. RGB, YBR_FULL, YBR_FULL_422, etc).
                    # Bits Allocated (0028,0100) defines how much space is allocated in the buffer for every sample in bits.
                    # 0x0028, 0x0101, 'US', "Bits Stored"
                    '''
                    #pdb.set_trace()
                    #if ds[0x0028, 0x0004].value == 'PALETTE_COLOR':
                    #    img = pydicom.pixel_data_handlers.util.apply_color_lut(img, ds)

                    if ds[0x0028, 0x0100].value == 16:
                        bit_depth = ds[0x0028, 0x0101].value
                        if bit_depth == 16:
                            img16 = img.astype(np.uint16)
                            img = (img16 / 256).astype(np.uint8)
                        elif bit_depth == 12:
                            img = (img / 16).astype(np.uint8)
                        else:
                            img = (img /(2**(bit_depth-8))).astype(np.uint8)
                    elif ds[0x0028, 0x0100].value == 8:
                        img = img.astype(np.uint8)
                    else:
                        raise Exception("unknown Bits Allocated value in dicom header")

                    img = img.reshape(img.shape[0], img.shape[1], -1)
                else:
                    img = cv2.imread(FilewithPath, cv2.IMREAD_COLOR)
                    #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            except:
                print("read image error, ignore the file")

            if img is None:
                cycle_cnt = cycle_cnt - 1
                time.sleep(0.02 * (repeat_time-cycle_cnt))

        if img is not None and img.shape[2] == 1:
            # repeat 3 times to make fake RGB images
            img = np.tile(img, [1, 1, 3])

    return img

def loadImageList(imglist_filepath):
    img_dir = imglist_filepath
    name_list = []
    X = []

    # Loop through the training and test folders, as well as the 'NORMAL' and 'PNEUMONIA' subfolders
    # and append all images into array X.  Append the classification (0 or 1) into array Y.
    #'''
    for fileName in os.listdir(img_dir):
        name_list.append(fileName)
        img = readImageFile(img_dir + fileName)
        if img is not None:
            X.append(img)

        # delete the physical image after read it
        time.sleep(0.01)
        cmd_str = "rm -f " + '"' + img_dir + fileName + '"'
        #print("delete input image file: ", cmd_str)
        os.system(cmd_str)

    return name_list, X

def apply_transforms(image, transforms):
    image = Image.fromarray(image)
    if transforms is not None:
        for trans in transforms:
            if type(trans) == ia.augmenters.meta.Sequential:
                image = trans.augment_images([np.array(image)])[0]
                image = Image.fromarray(image)
            else:
                image = trans(image)

    return image

def smart_handle_input_image(image):
    # pre-processing the image based on the image resolution
    # height, width, number of channels in image
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    if channels == 1:
        # repeat 3 times to make fake RGB images
        image = np.tile(image, [1, 1, 3])

    ratio = float(width/height)
    if (ratio < 0.90):
        # switch x and y
        image = image.transpose((1, 0, 2))
        ratio = float(width / height)

    if ratio>=1.35 and ratio<=1.65:
        image = apply_transforms(image, [data_transforms_test])
    elif ratio>1.65:
        # crop to 1.25
        width_resize = int(height * 1.25)
        image = cv2.resize(image, (height, width_resize))
        image = apply_transforms(image, [data_transforms_test_2])
    else:
        image = apply_transforms(image, [data_transforms_test_2])

    return image


def get_heatmap(model, img_path, class_val, resize=True, p=1):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # class_val = np.argmax(preds)
    african_elephant_output = model.output[:, class_val]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    # We use cv2 to load the original image
    if resize == True:
        img = cv2.resize(cv2.imread(img_path), (512, 512))
    else:
        img = cv2.imread(img_path)
    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img

    final_img = np.concatenate((img, superimposed_img), axis=1)
    # Save the image to disk
    if class_val == 0:
        cv2.imwrite(str(p) + '_' + str(class_val) + '_.jpg', final_img)
    elif class_val == 1:
        cv2.imwrite(str(p) + '_' + str(class_val) + '_.jpg', final_img)
    return


if __name__ == '__main__':
    if torch.cuda.is_available():
        #Sprint('torch.cuda.get_device_name=', torch.cuda.get_device_name())
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count()>1:
            torch.cuda.set_device(0)
            print('torch.torch.cuda.device_count()=', torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))


    if not os.path.exists(testdir):
        os.makedirs(testdir)
    if not os.path.exists(outputsdir):
        os.makedirs(outputsdir)

    start = time.time()

    batch_size = 1
    #models_list = ['ResNet18', 'ResNet50', 'DenseNet201']
    models_list = ['VI_CNN']
    data_transforms_test = transforms.Compose([
        # transforms.Resize(144),
        #transforms.ToPILImage(),
        transforms.Resize((600, 900)),
        #transforms.Resize((1000, 750)),
        transforms.CenterCrop((600, 750)),
        transforms.Resize((400, 500)),
        # transforms.Pad(16, padding_mode='reflect'),
        # transforms.CenterCrop(80),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_transforms_test_2 = transforms.Compose([
        transforms.Resize((400, 500)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    performance_list = []
    model_fts = []

    #for model_name in models_list:
    for i, model_name in enumerate(models_list):
        if model_name == 'VI_CNN':
            cp_filepath = 'best_model.h5'
            net = load_model(cp_filepath)
            print("Loaded checkpoint '%s'." % cp_filepath)

            model_fts.append(net)

            '''
            yt = model.predict(x3)
            print (roc_auc_score(y3,yt))

            for i in range(0,len(p)) :
                cv2.imwrite('test_gradient.jpg',x3[i])
                img_path = 'test_gradient.jpg'
                resize = False
            '''

    end_time = time.time()
    print('loading %d model, time %.2f' %(len(model_fts), end_time - start))

    timing = True
    save_result_to_csv = 1

    while timing:
        model_num = len(model_fts)
        num_class = 2
        input_path = glob.glob(testdir + '*')
        if len(input_path) > 0:
            error_code = 0
            predict_test = []
            label_test = []
            performance_list = []
            Y_prob = []

            # input image resolution is (224, 224, 3)
            filename, X = loadImageList(testdir)
            if len(X)<1:
                # image load error -- wrong image type or corrupted image data
                error_code = 1
                print("wrong input file: ", len(X))
            elif len(X)>1:
                print("input file number: ", len(X))

            # Note: do nothing if input image is invalid
            for imgs in X:
                # image is 1024x1024x3
                # VI training image is (600, 900, 3)
                # xiaofeng add for debug
                outfile = './input_test.png'
                im = Image.fromarray(cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
                im.save(outfile)

                for i in range(int(model_num)):
                    model_name = models_list[i]
                    model = model_fts[i]
                    # it relates to the output in own_models define
                    if model_name == 'VI_CNN':
                        #cls_probs = model_ft2.predict(imgs)
                        #cls_scores = net(im_data, False, torch.tensor(0), aux_feats)
                        #cls_probs = F.softmax(cls_scores, dim=-1)
                        orig_imgs = imgs.copy()
                        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                        imgs = cv2.resize(imgs, (224, 224))

                        class_val = 1
                        x = image.img_to_array(imgs)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)
                        preds = model.predict(x)

                        if preds[0][1] >= 0.0131:
                            class_val = 1
                        else:
                            class_val = 0

                        african_elephant_output = model.output[:, class_val]
                        last_conv_layer = model.get_layer('block5_conv3')
                        grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

                        # https://gist.github.com/haimat/10a53ad9675f8f5ac1290f06c3e4f973
                        # Get the gradient of the winner class with regard to the output of the (last) conv. layer
                        def _watch_layer(layer, tape):
                            def decorator(func):
                                def wrapper(*args, **kwargs):
                                    # Store the result of `layer.call` internally.
                                    layer.result = func(*args, **kwargs)
                                    # From this point onwards, watch this tensor.
                                    tape.watch(layer.result)
                                    # Return the result to continue with the forward pass.
                                    return layer.result

                                return wrapper

                            layer.call = decorator(layer.call)
                            return layer

                        # with tf.GradientTape() as gtape:
                        #     import pdb
                        #     pdb.set_trace()
                        #     _watch_layer(last_conv_layer, gtape)
                        #     # preds = model.predict(x)
                        #     # model_prediction = model.output[:, np.argmax(preds[0])]
                        #     grads = gtape.gradient(african_elephant_output, last_conv_layer.output)
                        # #end of the xiaofeng modification

                        pooled_grads = K.mean(grads, axis=(0, 1, 2))
                        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
                        pooled_grads_value, conv_layer_output_value = iterate([x])
                        for i in range(512):
                            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
                        heatmap = np.mean(conv_layer_output_value, axis=-1)
                        heatmap = np.maximum(heatmap, 0)
                        heatmap /= np.max(heatmap)
                        #plt.matshow(heatmap)
                        #plt.show()

                        # We use cv2 to load the original image -- we show a higher solution
                        #img = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
                        img = cv2.resize(orig_imgs, (512, 512)) #by default is RGB

                        # We resize the heatmap to have the same size as the original image
                        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                        # We convert the heatmap to RGB
                        heatmap = np.uint8(255 * heatmap)
                        # We apply the heatmap to the original image
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                        # 0.4 here is a heatmap intensity factor
                        superimposed_img = heatmap * 0.4 + img

                        final_img = np.concatenate((img, superimposed_img), axis=1)
                        # Save the image to disk
                        output_heatmap = './outputs/heatmap.jpg'
                        cv2.imwrite(output_heatmap, final_img)

                        cls_probs = preds
                        #cls_probs = cls_probs.data.numpy()
                        print("Softmax output: ", cls_probs)
                        Y_prob.append(cls_probs[0])
                        # xiaofeng: the threshold for future usage
                        #y_pred = (cls_probs[:, 1] >= cls_probs[:, 0]).astype(int)


            # input images have been processed, need to handle the output -- save them by model
            if save_result_to_csv == 1 and error_code != 0:
                B = open(outputsdir + 'error.csv', 'w')
                B.write('file name,error message\n')
                if error_code == 1:
                    errMsg = 'Wrong image file type or corrupted image data!'
                else:
                    errMsg = 'Other Error'
                B.write('{},{}\n'.format(filename[0], errMsg))
                B.close()

                B = open(outputsdir + 'predictions.csv', 'w')
                B.write(
                    'file name,Absence of Pathological Visual Impairment (Probability),probability of VI,probability of other abnormal,groud truth(0 = Normal  1 = VI),prediction(0 = Normal 1 = VI)\n')
                B.write('{},{},{},{},{},{}\n'.format(filename[0], -1, -1, "None", "None", -1))
                B.close()


            if len(Y_prob) > 0:
                Y_prob = np.array(Y_prob)
                #print(Y_prob.shape)

            if save_result_to_csv == 1 and error_code == 0:

                statistic_file = outputsdir + 'TestResult.csv'
                if os.path.isfile(statistic_file):
                    C = open(statistic_file, 'a+')
                else:
                    C = open(statistic_file, 'w')
                    C.write('file name,AI_model,model_output(VI),probability of Absence of VI,probability of Presence of VI,probability of other abnormal,groud truth(0 = Non-VI  1 = VI),prediction(0 = Non-VI 1 = VI)\n')

                B = open(outputsdir + 'predictions.csv', 'w')
                #B.write(
                #    'file name,probability of normal,probability of VI,probability of other abnormal,groud truth(0 = normal  1 = VI  2 = other abnormal), prediction(0 =  normal 1 = VI 2 = other abnormal)\n')
                #B.write('file name,Absence of Pathological Visual Impairment (Probability),Presence of Pathological Visual Impairment (Probability),probability of other abnormal,groud truth(0 = Normal  1 = VI),prediction(0 = Normal 1 = VI)\n')
                B.write(
                    'file name,Absence of VI (Probability),Presence of VI (Probability),probability of other abnormal,groud truth(0 = Normal  1 = VI),prediction(0 = Normal 1 = VI)\n')

                # the result is not important for VI
                Threshold_VI = [0.0131]
                y_value = np.zeros((num_class, model_num))
                pred_value = np.zeros(model_num)
                class_result = 0
                probability_0 = probability_1 = 0
                for ix in range(Y_prob.shape[0] // model_num):
                    # VI_CNN
                    if model_name == 'VI_CNN':
                        y_value = np.zeros(num_class)
                        for i in range(model_num):
                            y_value[0] += Y_prob[ix + i][0]
                            y_value[1] += Y_prob[ix + i][1]
                        y_value[0] = float(y_value[0] / model_num)
                        y_value[1] = float(y_value[1] / model_num)

                        if y_value[1] >= Threshold_VI[i]:
                            class_result = 1
                            probability_1 = 0.5 + ((y_value[1] - Threshold_VI[i]) / y_value[1]) * 0.5
                            probability_0 = 1- probability_1
                        else:
                            class_result = 0
                            probability_1 = 0.5 - ((Threshold_VI[i] - y_value[1]) / Threshold_VI[i]) * 0.5
                            probability_0 = 1- probability_1

                        pred_value[0] = y_value[1]
                        B.write('{},{},{},{},{},{}\n'.format(filename[ix], probability_0, probability_1, "None",
                                                             "None", class_result))
                        C.write('{},{},{},{},{},{},{},{}\n'.format(filename[ix], models_list[i], y_value[1], probability_0, probability_1, "None",
                                                                "None", class_result))

                B.close()
                C.close()

                for i in range(model_num):
                    print('[%s: Threshold(%.4f) / Prediction: %0.4f]  probability: Normal: %.4f; VI: %.4f'
                          %(models_list[i], Threshold_VI[i], pred_value[i], probability_0, probability_1))

        else:
            time.sleep(1.0)
            # print('.', end='')
