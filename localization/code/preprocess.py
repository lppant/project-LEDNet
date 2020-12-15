import os
import numpy as np
from skimage import io, exposure

def preprocess_lungs():
    jsrt_path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/JSRT/All247images/'
    preprocess_output_path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/JSRT/preprocess_output/'

    for i, filename in enumerate(os.listdir(jsrt_path)):
        if not filename.startswith('.'):
            print('doing lungs preprocess::' + filename)
            img = 1.0 - np.fromfile(jsrt_path + filename, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
            img = exposure.equalize_hist(img)
            io.imsave(preprocess_output_path + filename[:-4] + '.png', img)
            print('Lung', i, filename)

def preprocess_masks():
    jsrt_path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/JSRT/All247images/'
    preprocess_output_path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/JSRT/preprocess_output/'
    left_lungs_mask_path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/JSRT/Masks/left_lungs/'
    right_lungs_mask_path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/JSRT/Masks/right_lungs/'
    for i, filename in enumerate(os.listdir(jsrt_path)):
        if not filename.startswith('.'):
            print('doing masks preprocess::' + filename)
            left = io.imread(left_lungs_mask_path + filename[:-4] + '.gif')
            right = io.imread(right_lungs_mask_path + filename[:-4] + '.gif')
            io.imsave(preprocess_output_path + filename[:-4] + 'mask.png', np.clip(left + right, 0, 255))
            print('Mask', i, filename)

preprocess_lungs()
preprocess_masks()
