import numpy as np
from skimage import transform, io, img_as_float, exposure
import os

"""
Data was preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
    - normalize by data set mean and std.
Resulting shape should be (n_samples, img_width, img_height, 1).

It may be more convenient to store preprocessed data for faster loading.

Dataframe should contain paths to images and masks as two columns (relative to `path`).
"""

def loadDataJSRT(df, path, im_shape):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X, y = [], []
    for i, item in df.iterrows():
        img = io.imread(path + item[0])
        img = transform.resize(img, im_shape)
        img = np.expand_dims(img, -1)
        mask = io.imread(path + item[1])
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print '### Data loaded'
    print '\t{}'.format(path)
    print '\t{}\t{}'.format(X.shape, y.shape)
    print '\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max())
    print '\tX.mean = {}, X.std = {}'.format(X.mean(), X.std())
    return X, y


def loadDataGeneral(df, path, im_shape):
    """Function for loading arbitrary data in standard formats"""
    X, y = [], []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[0]))
        mask = io.imread(path + item[1])
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print '### Dataset loaded'
    print '\t{}'.format(path)
    print '\t{}\t{}'.format(X.shape, y.shape)
    print '\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max())
    print '\tX.mean = {}, X.std = {}'.format(X.mean(), X.std())
    return X, y


def loadChexpertDataForPrediction(df, path, im_shape):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    #X, y = [], []
    X = []
    for i, item in df.iterrows():
        img = io.imread(path + item[0])
        img = transform.resize(img, im_shape)
        img = np.expand_dims(img, -1)
        X.append(img)
    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    print '### Data loaded'
    print '\t{}'.format(path)
    print '\t{}'.format(X.shape)
    print '\tX:{:.1f}-{:.1f}\n'.format(X.min(), X.max())
    print '\tX.mean = {}, X.std = {}'.format(X.mean(), X.std())
    return X

def loadChexpertFromDir(batch, path, type, im_shape):

    print("Started Loading checkpert data for batch " + batch + " from " + os.path.join(path, type))
    X, file_paths = [], []
    file_count = 0
    for root, dirs, files in os.walk(os.path.join(path, type)):
        for file in files:
            patient_batch_name = 'patient' + batch
            if patient_batch_name in root and file.endswith("frontal.jpg"):
                file_path = os.path.join(root, file)
                print(file_path)
                img = io.imread(file_path)
                img = transform.resize(img, im_shape)
                img = np.expand_dims(img, -1)
                X.append(img)
                file_paths.append(file_path)
                file_count = file_count + 1
    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    print '### Data loaded'
    print '\t{}'.format(path)
    print('total files loaded: ' + str(file_count))
    print '\t{}'.format(X.shape)
    print '\tX:{:.1f}-{:.1f}\n'.format(X.min(), X.max())
    print '\tX.mean = {}, X.std = {}'.format(X.mean(), X.std())
    return X, file_paths
