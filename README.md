# project-LEDNet

## Important Note:
* Model files were too large hence they were removed from code submission 
* Model Files are committed at : 
    * classification/model_originial/densenet.pth -- Original CheXpert dataset
    * classification/model_localized/densenet.pth -- Localized dataset
    * localization/code/trained_model.hdf5 -- Pre-trained model
    * localization/code/model.009.hdf5 -- Self-trained model

## Localization
Localization contains code for extracting localized lung region images from the input chest X-ray images.

### Citiation
*   We have used this [U-Net implementation] (https://github.com/imlab-uiip/lung-segmentation-2d) and modified their code where needed for data load or prediction.
*   We have also used their pre-trained model for prediction purposes. 
 
### Pre-requisites
*   Setup the conda environment using `conda env create <environment yml file>`
*   To run on CPU, use `code/env_cpu/environment.yml` as environment yml file
*   To run on GPU, use `code/env_gpu/environment.yml` as environment yml file

### Data
*   [JSRT dataset](http://db.jsrt.or.jp/eng.php) for 247 chest-Xray images
*   Corresponding left and right lung region masks from [SCR database](https://www.isi.uu.nl/Research/Databases/SCR/)
*   [Chexpert images](https://stanfordmlgroup.github.io/competitions/chexpert/) to predict the lung region masks using the model
*   Above links should be used to download the data and provide the path to downloaded data in the code before trigerring a run

### Usage
*   Run `code/preprocess.py` :
    * to perform histrogram equalization on JSRT chest-Xray images
    * to combine left and right lung masks into single image
    * replace `jsrt_path` variable with [JSRT dataset](http://db.jsrt.or.jp/eng.php) path
    * replace `left_lungs_mask_path` with path of left lung mask images from [SCR database](https://www.isi.uu.nl/Research/Databases/SCR/)
    * replace `right_lungs_mask_path` with path of right lung mask images from [SCR database](https://www.isi.uu.nl/Research/Databases/SCR/)
    * replace `preprocess_output_path` variable with pre-processing output directory. This directory should be created before running the code.
*   Run `code/train_model.py` to train the model for generating lung masks using U-Net implementation with:
    * preprocessed JSRT chest-Xray images as X (input vector)
    * preprocessed single image for left and right lung masks as Y (output vector)
    * To run the file, replace `path` variable with [JSRT dataset](http://db.jsrt.or.jp/eng.php) path
*   Run `code/inference.py` to use the model for generating lung masks from Chexpert Images. To run the file:
    * replace `path` variable with [Chexpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) path
    * set `batch` variable value as 'train' or 'valid'
    * best perfoming model is commited as file named `model.009.hdf5`
    * pre-trained model is commited as file named `trained_model.hdf5`

## Classification
Classification contains code for predicting diseases from images with labels.

### Pre-requisites
*   Setup the conda environment using `conda env create <environment yml file>`
*   Where environment file is, `code/environment.yml`

### Data
*   [Chexpert overlay dataset](https://drive.google.com/open?id=11nQVVnzN3quw2c5cjhAtldHn7YbNFndX) containing 28,929 chest X-ray images
*   [Chexpert original dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) corresponding to the above 28,929 overlay images
*   Above links should be used to download the data and provide the path to downloaded data in the code before trigerring a run

### Usage
*   Run `code/python etl_chexpert_data.py -h` to get full detailed command
usage: etl_chexpert_data.py [-h] -c CSV_PATH -p PREFIX_PATH -d DEST_PATH -o
                            OVERLAY

divide the data in train validate and test

optional arguments:
  -h, --help      show this help message and exit
  -c CSV_PATH     Path to file containing file name and labels
  -p PREFIX_PATH  Path to directory containing image dataset
  -d DEST_PATH    Path to output directory
  -o OVERLAY      Is overlay images Y for yes N for no!

*   Run `code/python train_densenet.py -h`
usage: train_densenet.py [-h] -p PREFIX_PATH

train densenet 121 with 9 epocs and batch size of 50

optional arguments:
  -h, --help      show this help message and exit
  -p PREFIX_PATH  Path to directory containing image dataset

