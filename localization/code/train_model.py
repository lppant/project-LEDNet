from image_gen import ImageDataGenerator
from load_data import loadDataJSRT
from build_model import build_UNet2D_4L

import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = 'idx_jsrt.csv'

    path = '/Users/pantla/Documents/Study/OMSCS/Subjects/BD4H/project/data/JSRT/preprocess_output/'

    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1, random_state=23)
    n_train = int(len(df) * 0.8)
    print('Im n_train###' + str(n_train))
    df_train = df[:n_train]
    print('df_train size: ' + str(len(df_train)))
    df_val = df[n_train:]
    print('df_val size: ' + str(len(df_val)))

    # Load training and validation data
    im_shape = (256, 256)
    X_train, y_train = loadDataJSRT(df_train, path, im_shape)
    X_val, y_val = loadDataJSRT(df_val, path, im_shape)

    # Build model
    inp_shape = X_train[0].shape
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    # plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    model_file_format = 'model.{epoch:03d}.hdf5'
    print model_file_format
    checkpointer = ModelCheckpoint(model_file_format, period=1)

    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)
    print('completed train_gen')
    test_gen = ImageDataGenerator(rescale=1.)
    print('Starting model fitting')
    batch_size = 8
    UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=10,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)
    print('Done')
