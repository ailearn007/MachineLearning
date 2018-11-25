#coding = utf-8
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras import initializers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.optimizers import Adam

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def normalize_data(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes):
    num_class = len(classes)
    X_train = train_set_x_orig.reshape(-1, 64,64,3)/255
    X_test = test_set_x_orig.reshape(-1, 64, 64, 3)/255

    Y_train = np.squeeze(np_utils.to_categorical(train_set_y_orig, num_classes= num_class))
    Y_test = np.squeeze(np_utils.to_categorical(test_set_y_orig, num_classes= num_class))
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes =\
    load_dataset()

    # X_train, Y_train, X_test, Y_test = \
    #     normalize_data(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes)
    X_train = train_set_x_orig/255
    Y_train = np.squeeze(np_utils.to_categorical(train_set_y_orig.T, num_classes= 6))

    X_test = test_set_x_orig/255
    Y_test = np.squeeze(np_utils.to_categorical(test_set_y_orig.T, num_classes= 6))


    model = Sequential()

    np.random.seed(12)
    model.add(Convolution2D(
        batch_input_shape= (None, 64, 64, 3),
        filters= 6,
        kernel_size= (5,5),
        strides= (2,2),
        padding= 'valid',
        kernel_initializer= initializers.he_normal()
        #data_format= 'channels_first',
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(
        pool_size= (2,2),
        strides= (2,2),
        padding= 'valid',
        #data_format= 'channels_first'
    ))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(16, 5, strides= 1, padding= 'valid',
                            kernel_initializer= initializers.he_normal()
                            #data_format= 'channels_first'
                            ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(2, 2, 'valid',
                           #data_format= 'channels_first'
                           ))

    model.add(Flatten())
    model.add(Dense(120, kernel_initializer= initializers.he_normal()))
    model.add(Activation('relu'))

    model.add(Dense(84, kernel_initializer= initializers.he_normal()))
    model.add(Activation('relu'))

    model.add(Dense(6, kernel_initializer= initializers.he_normal()))
    model.add(Activation('softmax'))

    adam = Adam(lr = 0.01)

    model.compile(optimizer= adam,
                  loss= 'categorical_crossentropy',
                  metrics= ['accuracy'])

    model.fit(X_train, Y_train, epochs= 30, batch_size= 64, verbose= 2)
    model.summary()

    loss, accuracy = model.evaluate(X_train, Y_train, verbose=0)
    print(loss, accuracy)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose= 0)
    print(loss, accuracy)

