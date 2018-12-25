#coding = utf-8
import time
import numpy as np
from keras.layers import SimpleRNN, Input, Dense, Activation
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras import callbacks

def dataLoad_2(filename):
    data = open(filename, 'r').read()
    data = data.lower()
    chars =  list(set(data))
    char_to_ix = {ch:(i+1) for i,ch in enumerate(sorted(chars))}
    ix_to_char = {(i+1):ch for i,ch in enumerate(sorted(chars))}
    #print(ix_to_char)
    #print(char_to_ix)
    #data_size, vocab_size = len(data), len(chars)
    #print('There are {0:d} total characters and {1:d} unique characters in your data.'\
     #     .format(data_size, vocab_size))
    return data, char_to_ix, ix_to_char

def form_train_X(filename = 'dinos.txt'):
    data, char_to_ix, ix_to_char = dataLoad_2(filename)

    with open(filename) as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    lens = [len(example) for example in examples]

    Tx = max(lens)
    m = len(examples)
    features = len(char_to_ix) + 1
    train_X = np.zeros((m, Tx, features))

    for i in range(m):
        example = examples[i]
        x_int = [char_to_ix[char] for char in example]
        train_X[i,:len(x_int),:] = to_categorical(x_int, features)

    return train_X

if __name__  == '__main__':
    train_X = form_train_X()

    inputs = Input(train_X.shape[1:])
    X = SimpleRNN(train_X.shape[1], return_sequences= True, unroll= True)(inputs)
    X = Dense(train_X.shape[2])(X)
    X = Activation('softmax')(X)

    model = Model(input = inputs, output = X)
    model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])

    t1 = time.time()
    model.fit(train_X, train_X, epochs= 85, batch_size= 256, verbose= 1, callbacks=[callbacks.ProgbarLogger()])
    t2 = time.time()

    model.summary()
    print(t2 - t1, 's')
