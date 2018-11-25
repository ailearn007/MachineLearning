# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

def dataLoad(filename):
    data = open(filename, 'r').read()
    data = data.lower()
    chars =  list(set(data))
    char_to_ix = {ch:i for i,ch in enumerate(sorted(chars))}
    ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}
    print(ix_to_char)
    print(char_to_ix)
    data_size, vocab_size = len(data), len(chars)
    print('There are {0:d} total characters and {1:d} unique characters in your data.'\
          .format(data_size, vocab_size))
    return data, char_to_ix, ix_to_char

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum(axis = 0)

def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)  #可不加
    parameters = {}
    parameters['Wax'] = np.random.randn(n_a, n_x)*0.01
    parameters['Waa'] = np.random.randn(n_a, n_a)*0.01
    parameters['ba'] = np.zeros((n_a, 1))
    parameters['Wya'] = np.random.randn(n_y, n_a)*0.01
    parameters['by'] = np.zeros((n_y, 1))
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    '''
    compute one step forward at time step: t
    :param parameters: dict
    :param a_prev: a<t-1>
    :param x: x<t>
    :return: a: a<t>   y: y<t>
    '''
    Wax, Waa, ba, Wya, by = parameters['Wax'], parameters['Waa'], \
                            parameters['ba'], parameters['Wya'],parameters['by']
    a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
    y = softmax(np.dot(Wya, a) + by)
    return a, y

def rnn_forward(X, Y, a0, parameters, vocab_size):
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if X[t] != None:
            x[t][X[t]] = 1
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        loss -= np.log(y_hat[t][Y[t], 0])

    cache = (y_hat, a, x)
    return loss, cache

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    '''
    compute gradients in time step: t
    :param dy: 2D ndarray (n_y, 1): dy<t> =y_hat<t> - y<t> is dz<t>
    :param gradients:  dict: dWya dby dWaa dWax dba da_next
    :param parameters:  dict: Wya by Waa Wax ba
    :param x: 2D ndarray (n_x, 1): x<t>
    :param a: 2D ndarray (n_a, 1): a<t>
    :param a_prev: 2D ndarray (n_a, 1): a<t-1>
    :return: gradients
    '''
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] #da_next is da<t> from time step: t+1
    daraw = (1 - a*a)*da #tanh derivative
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['dba'] += daraw
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw) #da_next is da<t-1> passing to time step: t-1
    return gradients


def rnn_backward(X, Y, parameters, cache):
    gradients = {}
    (y_hat, a, x) = cache
    Wax, Waa, ba, Wya, by = parameters['Wax'], parameters['Waa'], \
                            parameters['ba'], parameters['Wya'],parameters['by']

    gradients['dWax'], gradients['dWaa'], gradients['dba'], gradients['dWya'], gradients['dby'] = \
    np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(ba), np.zeros_like(Wya), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t]) #y_hat<t>
        dy[Y[t]] -= 1 # y_hat<t> - y<t>
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])

    return gradients, a

def update_parameters(parameters, gradients, lr):
    parameters['Wax'] -= lr*gradients['dWax']
    parameters['Waa'] -= lr*gradients['dWaa']
    parameters['ba'] -= lr*gradients['dba']
    parameters['Wya'] -= lr*gradients['dWya']
    parameters['by'] -= lr*gradients['dby']
    return parameters

def clip(gradients, maxValue):
    dWaa, dWax, dba, dWya, dby = gradients['dWaa'], gradients['dWax'], gradients['dba'], \
                                 gradients['dWya'], gradients['dby']
    for gradient in [dWaa, dWax, dba, dWya, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient)

    gradients = {'dWaa': dWaa, 'dWax': dWax, 'dba': dba, 'dWya': dWya, 'dby': dby}
    return gradients

def optimize(X, Y, a_prev, parameters, vocab_size, learning_rate):
    loss, cache = rnn_forward(X, Y, a_prev, parameters, vocab_size)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, maxValue = 5)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, parameters, gradients, a[len(X) - 1]



def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length

def smooth(loss, cur_loss):
    return loss*0.999 + cur_loss*0.001

def model(char_to_ix, num_iter = 35000, n_a = 50, dino_names = 7, vocab_size = 27, learning_rate = 0.01):
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)

    with open('dinos.txt') as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    np.random.seed(0)
    np.random.shuffle(examples)

    losses = []
    a_prev = np.zeros((n_a, 1))

    for j in range(num_iter):
        index = j%len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix['\n']]

        cur_loss, parameters, gradients, a_prev = \
            optimize(X, Y, a_prev, parameters, vocab_size, learning_rate)
        loss = smooth(loss, cur_loss)

        if j%2000 == 0:
            losses.append(loss)
            print('Iter: {0}, Loss: {1:8.6}'.format(j, loss))

    plt.plot(np.squeeze(losses))
    plt.xlabel('iterations per 2000')
    plt.ylabel('losses')
    plt.title('Learning rate = {0:^5}'.format(learning_rate))
    plt.show()
    return parameters






if __name__ == '__main__':
    filename = 'dinos.txt'
    data, char_to_ix, ix_to_char = dataLoad(filename)
    parameters = model(char_to_ix, \
                       num_iter = 35000, n_a = 50, dino_names = 7, vocab_size = 27, learning_rate = 0.01)

