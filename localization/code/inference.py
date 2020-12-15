from load_data import loadDataJSRT, loadChexpertDataForPrediction, loadChexpertFromDir

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure
from tensorflow.python.client import device_lib
from keras import backend as K
from pathlib import Path
import os

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    #boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    color_mask[mask == 1] = [0, 0, 1]
    #color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def save_mask_and_overlay(file_path, img, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [1, 1, 1]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    img_overlay = np.multiply(img_masked, color_mask)
    io.imsave('{}'.format(file_path[:-4] + '_overlay_img.jpg'), img_overlay)
    return

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':

    #print('Device Lib')
    #print(device_lib.list_local_devices())

    #print('Get Available GPUs')
    #K.tensorflow_backend._get_available_gpus()

    path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/CheXpert-v1.0-small/'

    # Load test data
    im_shape = (256, 256)
    #batches = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59']
    batches = ['']
    type = 'valid'
    for batch in batches:
        X, file_paths = loadChexpertFromDir(batch, path, type, im_shape)
        n_test = X.shape[0]
        print('n_test::' + str(n_test))

        inp_shape = X[0].shape
        print('inp_shape::' + str(inp_shape))
        # Load model
        model_name = 'trained_model.hdf5'
        UNet = load_model(model_name)

        # For inference standard keras ImageGenerator is used.
        test_gen = ImageDataGenerator(rescale=1.)

        ious = np.zeros(n_test)
        dices = np.zeros(n_test)

        i = 0
        for xx in test_gen.flow(X, batch_size=1):
            img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
            pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
            #mask = yy[..., 0].reshape(inp_shape[:2])

            # Binarize masks
            #gt = mask > 0.5
            pr = pred > 0.5

            # Remove regions smaller than 2% of the image
            pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
            mask_path = file_paths[i].replace(type, type + '_mask')
            parent_mask_path = Path(mask_path).parent
            if not os.path.exists(str(parent_mask_path)):
                os.makedirs(str(parent_mask_path))

            save_mask_and_overlay(mask_path, img, pr)
            print('saved masked file:::' + mask_path)
            print('Completed ' + str(i+1))

            #ious[i] = IoU(gt, pr)
            #dices[i] = Dice(gt, pr)
            #print df.iloc[i][0], ious[i], dices[i]

            i += 1
            if i == n_test:
                break

    #print 'Mean IoU:', ious.mean()
    #print 'Mean Dice:', dices.mean()

