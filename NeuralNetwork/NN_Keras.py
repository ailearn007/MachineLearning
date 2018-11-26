# coding = utf-8
import numpy as np
from keras.layers import Dense, Input, Activation, initializers
from keras.models import Sequential, Model

def load_nn_data():
    '''
    加载数据
    :return: train_X ndarray二维 (n_x, m)； train_Y ndarray二维 (1,m)
    '''
    train_X = np.load('DataSet/train_X.npy')
    train_Y = np.load('DataSet/train_Y.npy')
    test_X = np.load('DataSet/test_X.npy')
    test_Y = np.load('DataSet/test_Y.npy')
    return train_X, train_Y, test_X, test_Y

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_nn_data()
    print(train_X.shape, train_Y.shape)
    print(train_X.shape[0:1])
    print(train_X.shape[0])

    model = Sequential()
    model.add(Dense(8,
                    input_dim= train_X.shape[0],
                    #input_shape=(train_X.shape[0:1]),
                    kernel_initializer= initializers.he_normal()))
    model.add(Activation('relu'))

    model.add(Dense(4, kernel_initializer= initializers.he_normal()))
    model.add(Activation('relu'))

    model.add(Dense(3, kernel_initializer= initializers.he_normal()))
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer= initializers.he_normal()))
    model.add(Activation('sigmoid'))

    model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
    model.fit(train_X.T, train_Y.T, batch_size= 64, epochs=500, verbose= 2)
    loss, accuracy = model.evaluate(test_X.T, test_Y.T, verbose= 2)
    print(loss, accuracy)

