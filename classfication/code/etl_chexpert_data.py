import os
import pandas as pd
from shutil import copyfile
import argparse


def get_arguments():
    '''
    Get the arguments for etl
    :return:
    '''
    parser = argparse.ArgumentParser(description="divide the data in train validate and test")
    parser.add_argument("-c", required=True, dest="csv_path",help="Path to file containing file name and labels")
    parser.add_argument("-p", required=True, dest="prefix_path", help="Path to directory containing image dataset")
    parser.add_argument("-d", required=True, dest="dest_path", help="Path to output directory")
    parser.add_argument("-o",required=True,dest="overlay",help="Is overlay images Y for yes N for no!")
    return parser.parse_args()

'''
#PATH_INPUT_CSV = "~/Downloads/CheXpert-v1.0-small/valid.csv"
#PATH_INPUT_DIR = "/Users/shubhamarora/study/CS-6250/projectv1/"
#PATH_OUT_DATASET_TRAIN = "../data_overlay/train/"
PATH_OUT_DATASET_VAL = "../data_overlay/val/"
PATH_OUT_DATASET_TEST = "../data_overlay/test/"
TRAIN_RATIO = .70
TEST_RATIO = .10
VAL_RATIO = .20

os.makedirs(PATH_OUT_DATASET_TRAIN, exist_ok=True)
os.makedirs(PATH_OUT_DATASET_VAL, exist_ok=True)
os.makedirs(PATH_OUT_DATASET_TEST, exist_ok=True)
'''



def main():
    args = get_arguments()
    PATH_INPUT_CSV = args.csv_path
    PATH_INPUT_DIR = args.prefix_path
    PATH_OUT_DATASET_TRAIN = args.dest_path + "/train/"
    PATH_OUT_DATASET_VAL = args.dest_path + "/val/"
    PATH_OUT_DATASET_TEST = args.dest_path + "/test/"
    os.makedirs(PATH_OUT_DATASET_TRAIN, exist_ok=True)
    os.makedirs(PATH_OUT_DATASET_VAL, exist_ok=True)
    os.makedirs(PATH_OUT_DATASET_TEST, exist_ok=True)
    TRAIN_RATIO = .70
    TEST_RATIO = .10
    VAL_RATIO = .20
    df = pd.read_csv(PATH_INPUT_CSV)
    df_frontal_imgs = df[df['Frontal/Lateral']=='Frontal'] #filter to have only frontal images
    cleansed_df = df_frontal_imgs.drop(['Sex','Age','Frontal/Lateral','AP/PA'],axis=1)
    cleansed_df['separated_val'] = cleansed_df.apply(lambda x : ','.join(x[1:].astype(str).values),axis=1)
    total_files = len(cleansed_df)
    train_files = int(TRAIN_RATIO * total_files)
    test_files = train_files + int(TEST_RATIO * total_files) - 1
    val_files = test_files + int(VAL_RATIO * total_files) - 1

    with open(PATH_OUT_DATASET_TRAIN + '/train.csv','w') as train,open(PATH_OUT_DATASET_VAL + '/val.csv','w') as val,\
            open(PATH_OUT_DATASET_TEST + '/test.csv','w') as test:
        for index,row in cleansed_df.iterrows() :
            if index < train_files :
                if(args.overlay == 'Y'):
                    copyfile(PATH_INPUT_DIR + row['Path'].replace(".jpg","_overlay_img.jpg"), PATH_OUT_DATASET_TRAIN + "/" + str(index) + ".jpg")
                else :
                    copyfile(PATH_INPUT_DIR + row['Path'], PATH_OUT_DATASET_TRAIN + "/" + str(index) + ".jpg")
                train.write(str(index) + ".jpg," + row['separated_val'] + "\n")
            if index > train_files and index < test_files :
                if (args.overlay == 'Y'):
                    copyfile(PATH_INPUT_DIR + row['Path'].replace(".jpg","_overlay_img.jpg"), PATH_OUT_DATASET_TEST + "/"+ str(index) + ".jpg")
                else:
                    copyfile(PATH_INPUT_DIR + row['Path'], PATH_OUT_DATASET_TEST + "/" + str(index) + ".jpg")
                test.write(str(index) + ".jpg," + row['separated_val'] + "\n")
            if index > test_files and index < val_files :
                if (args.overlay == 'Y'):
                    copyfile(PATH_INPUT_DIR + row['Path'].replace(".jpg","_overlay_img.jpg"), PATH_OUT_DATASET_VAL + "/"+ str(index) + ".jpg")
                else:
                    copyfile(PATH_INPUT_DIR + row['Path'], PATH_OUT_DATASET_VAL + "/" + str(index) + ".jpg")
                val.write(str(index) + ".jpg," + row['separated_val'] + "\n")


    #print(df_no_finding['Path'])

if __name__ == '__main__':
    main()
