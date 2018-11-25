#coding = utf-8
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Model, load_model, model_from_json, model_from_yaml
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D
from keras.optimizers import Adam
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras import initializers

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

if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes =\
    load_dataset()
    X_train = train_set_x_orig/255
    Y_train = np.squeeze(np_utils.to_categorical(train_set_y_orig.T, num_classes= 6))

    X_test = test_set_x_orig/255
    Y_test = np.squeeze(np_utils.to_categorical(test_set_y_orig.T, num_classes= 6))
    #print(X_train.shape[1:])

    np.random.seed(12)
    X_input = Input(X_train.shape[1:])
    X = Conv2D(6, (5,5), strides=(2,2), padding= 'valid', name = 'CONV_1', kernel_initializer=
               initializers.he_normal())(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides= (2,2), name= 'MAX_POOL_1')(X)

    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(16, (5,5), strides= (1,1), padding= 'valid', name= 'CONV_2', kernel_initializer=
               initializers.he_normal())(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides= (2,2), name= 'MAX_POOL_2')(X)

    X = Flatten()(X)
    X = Dense(120, activation= 'relu', name= 'FC_3',kernel_initializer=
               initializers.he_normal())(X)
    X = Dense(84, activation= 'relu', name= 'FC_4', kernel_initializer=
               initializers.he_normal())(X)
    X = Dense(6, activation= 'softmax', name= 'FC_5', kernel_initializer=
               initializers.he_normal())(X)

    model = Model(inputs = X_input, outputs = X, name= 'LeNet5_K2')

    adam = Adam(lr = 0.01)
    model.compile(optimizer= adam, loss= 'categorical_crossentropy', metrics= ['accuracy'])

    model.fit(x= X_train, y = Y_train, epochs= 30, batch_size= 64, verbose= 2)

    loss, accuracy = model.evaluate(X_train, Y_train)
    print(loss, accuracy)
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(loss, accuracy)

    model.summary()
    # pydot与graphviz 安装有问题！
    # pydot1.1.0版本支持grapgviz，而pydot1.1.0与python3.6不兼容
    # plot_model(model, to_file= 'LeNet5_K2.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))

    # 保存模型
    # model.save('my_LeNet_5.h5')
    # model_1 = load_model('my_LeNet_5.h5')
    # loss, accuracy = model_1.evaluate(X_test, Y_test)
    # print('load_model',loss, accuracy)
    #
    # json_string = model.to_json()
    # model_2 = model_from_json(json_string)
    # model_2.compile(optimizer= adam, loss= 'categorical_crossentropy', metrics= ['accuracy'])
    # model_2.fit(x= X_train, y = Y_train, epochs= 30, batch_size= 64, verbose= 0)
    # loss, accuracy = model_2.evaluate(X_test, Y_test)
    # print('json',loss, accuracy)
    #
    # yaml_string = model.to_yaml()
    # model_3 = model_from_yaml(yaml_string)
    # model_3.compile(optimizer= adam, loss= 'categorical_crossentropy', metrics= ['accuracy'])
    # model_3.fit(x= X_train, y = Y_train, epochs= 30, batch_size= 64, verbose= 0)
    # loss, accuracy = model_3.evaluate(X_test, Y_test)
    # print('yaml',loss, accuracy)
    #
    # model.save_weights('my_LeNet_5_weights.h5')
    # model.load_weights('my_LeNet_5_weights.h5')
    # model.load_weights('my_LeNet_5_weights.h5', by_name= True)
