import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
# from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from oldData import *


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


class myUnet(object):
    def __init__(self, img_rows=100, img_cols=100):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        concat_axis = 3

        conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(
            inputs)
        conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
        conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
        conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

        conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
        conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

        conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
        conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

        conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
        conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

        up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
        conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

        up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
        conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

        up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
        conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

        up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
        conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
        conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model

        # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print
        # "conv1 shape:", conv1.shape
        # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # print
        # "conv1 shape:", conv1.shape
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # print
        # "pool1 shape:", pool1.shape
        #
        # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        # print
        # "conv2 shape:", conv2.shape
        # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # print
        # "conv2 shape:", conv2.shape
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # print
        # "pool2 shape:", pool2.shape
        #
        # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        # print
        # "conv3 shape:", conv3.shape
        # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # print
        # "conv3 shape:", conv3.shape
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # print
        # "pool3 shape:", pool3.shape
        #
        # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.5)(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        #
        # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # drop5 = Dropout(0.5)(conv5)
        #
        # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(drop5))
        # merge6 = concatenate([drop4, up6], axis=3)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        #
        # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv6))
        # merge7 = concatenate([conv3, up7], axis=3)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv7))
        #
        # merge8 = concatenate([conv2, up8], axis=3)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        #
        # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv8))
        # merge9 = concatenate([conv1, up9], axis=3)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        #
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        # model = Model(input=inputs, output=conv10)
        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        # return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=10, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('results/imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("results/%d.jpg" % (i))
            img.save("results/%d.jpg" % (i))


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()
