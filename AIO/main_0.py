# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:26:27 2021

@author: Leo
"""

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
# tf.compat.v1.disable_eager_execution()

m = 70
N = 100
Nd = 7
p = 10**2
k = 4  # number of active users
dv = 3
alpha = 10*N
bits = np.random.randint(2, size=[N, Nd])

# Wrong codebook, the output is bold C instead of bold C(t)
# def Codebook_gen(N, m, Nd, dv):
#     """
#     Parameters
#     ----------
#     m : int. Number of resource blocks
#     N : int. Number of users
#     dv : int. Number of chips that each user can occupy

#     Returns codebook matrix for all user
#     -------
#     """
#     Codebook = np.zeros((m, N), dtype=int)
#     # codebook matrix
#     index = np.zeros((dv, N), dtype=int)
#     index_list = []
#     for i in range(N):
#         index[:, i] = np.random.choice(m, dv, replace=False)
#         index[:, i] = np.sort(index[:, i])
#         index_list.append(index[:, i].tolist())
#         for j in range(i):
#             while index_list[j] == index_list[i]:
#                 index_list[j] = np.sort(np.random.choice(
#                     m, dv, replace=False)).tolist()

#     index = np.array(index_list).T
#     Codebook[index, np.arange(N)] = 1
#     # let the element be 1 by the order of
#     # [index[0][0], 0], [index[1][0], 1]... [index[end][0], end]
#     # [index[0][1], 0], [index[1][1], 1]... [index[end][1], end]
#     return(Codebook)


def Codebook_gen(N, m, Nd, dv):
    """
    Parameters
    ----------
    m : int. Number of resource blocks
    N : int. Number of users
    dv : int. Number of chips that each user can occupy

    Returns codebook matrix for all user
    -------
    """
    # Index generation
    index = np.zeros((dv, N), dtype=int)
    index_list = []
    for i in range(N):
        index[:, i] = np.random.choice(m, dv, replace=False)
        index[:, i] = np.sort(index[:, i])

        index_list.append(index[:, i].tolist())
        for j in range(i):
            while index_list[j] == index_list[i]:
                index_list[j] = np.sort(np.random.choice(
                    m, dv, replace=False)).tolist()
    Codebook_temp = np.zeros([m, N], dtype=int)
    Codebook_temp[index, np.arange(N)] = 1
    # Codebook in eq(1)

    # New method: assign to 1
    Codebook_new = np.zeros([m, m*N*Nd], dtype=int)
    start = time.process_time_ns()
    for i in range(np.shape(index)[0]):
        for j in range(np.shape(index)[1]):
            for k in range(Nd):
                Codebook_new[index[i, j], (index[i, j] + k*m + m*Nd*j)] = 1
    end = time.process_time_ns()
    # print('Old method time = ' + str(end - start) + 's')

    Codebook_list = []
    start1 = time.process_time_ns()
    for i in range(N):
        Codebook_Diag = np.diag(Codebook_temp[:, i])
        Codebook_list.append(np.tile(Codebook_Diag, [1, 7]))
    Codebook = np.hstack(Codebook_list)
    end1 = time.process_time_ns()
    # print('New method time = ' + str(end1 - start1) + 's')

    # Other method to create codebook
# =============================================================================
#     # Old method: concatenate codebook
#     Codebook = np.array([], dtype=int).reshape(m, 0)
#     for i in range(N):
#         Codebook_Diag = np.diag(Codebook_temp[:, i])
#         Codebook = np.concatenate(
#             (Codebook, np.tile(Codebook_Diag, (1, Nd))), axis=1)
#     # Upper bold phi in (5) without sub index
#
#     # Old method 2
#     a = np.arange(7)*70
#     a = np.tile(a, [dv, 1])
#     one_matrix = np.ones_like(a)
#     Codebook = np.zeros([m, m*N*Nd], dtype=int)
#     start1 = time.process_time_ns()
#     for i in range(np.shape(index)[1]):
#         b = a + one_matrix*i*m*Nd + np.tile(np.expand_dims(index[:, i], axis=1), [1, Nd])
#         Codebook[np.tile(np.expand_dims(index[:, i], axis=1), [1, Nd]), b] = 1
#     end1 = time.process_time_ns()
#     print('New method 2 time = ' + str(end1 - start1) + 's')
#     # Upper bold phi in (5) without sub index
# =============================================================================

    # Check
# =============================================================================
#     # New method check
#     # check = np.where((Codebook == Codebook_new) == False)
#     # if check[0].size == 0:
#     #     print('New method correct')
#     # else:
#     #     print('New method wrong')
#
#     # Codebook position check
#     # for i in range(5):
#     #     test = np.random.randint(0, 100, size=1)
#     #     test = test[0]  # to int
#     #     if (Codebook[index_list[test][0], index_list[test][0]+m*Nd*test] == 1) and (
#     #             Codebook[index_list[test][1], index_list[test][1]+m*Nd*test] == 1):
#     #         ans = "Correct"
#     #     else:
#     #         ans = "Failure"
#     #     print("Codebook test: " + ans)
# =============================================================================
    time1 = end1 - start1
    time0 = end - start
    return Codebook_new, time0, time1


# temp_old=0
# temp_new=0
# for i in range(100):
#     _, time0, time1 = Codebook_gen(N, m, Nd, dv)
#     temp_old = temp_old + time0
#     temp_new = temp_new + time1
# print(temp_old/100)
# print(temp_new/100)


# Useless, it's q(t) in eq.(4) instead of x in eq.(5)
def q_gen(N, m, Nd, delta):
    """
    Parameters
    ----------
    N : int. Number of users
    m : int. Number of resource blocks
    Nd : int. Number of data measurements
    delta : int. Active device indicator function

    Returns bold q(t)
    -------
    """
    # delta: device active indicator
    # index = np.random.choice(N, k, replace=False)
    # delta = np.zeros(N)
    # delta[index] = 1
    q = np.zeros((N*m, Nd), dtype='complex128')
    for j in range(Nd):
        i = 0
        while i < N:
            if delta[i] == 1:
                bits = np.random.randint(0, 2, size=[1, ])*2-1
                channel = np.random.randn(m, 1) + 1j*np.random.randn(m, 1)
                x = np.multiply(bits, channel)
                x = x.reshape(m,)
                # x = x.reshape(m, 1)
                q[m*i:m*i+m, j] = x
            i += 1
    q = np.reshape(q.T, (m*N*Nd))
    return q


def y_tilde_gen(Codebook, N, m, Nd, delta):
    q = q_gen(N, m, Nd, delta)
    y_tilde = np.dot(Codebook, q)
    y_tilde = np.real(np.hstack((y_tilde, np.imag(y_tilde)))).T
    y_tilde = np.expand_dims(y_tilde, axis=0)
    # copy imaginary part to the end of the vector
    return y_tilde
    # y_tilde.shape=(2m, 1) to concatenate


def train_data_gen(N, m, Nd, dv, p, k):
    delta_matrix = np.zeros((N, 1))
    y_hat_p = np.zeros((2*m, 1))
    for i in range(p):
        print(i)
        # initial
        index = np.random.choice(N, k, replace=False)
        delta = np.zeros((N, 1), dtype=int)
        delta[index] = 1
        # generation
        Codebook = Codebook_gen(N, m, Nd, dv)
        y_tilde = y_tilde_gen(Codebook, N, m, Nd, delta).T
        # concatenate
        y_hat_p = np.concatenate((y_hat_p, y_tilde), axis=1)
        delta_matrix = np.concatenate((delta_matrix, delta), axis=1)
    return y_hat_p[:, 1::], delta_matrix[:, 1::]


def Hidden_Layer(input_tensor, alpha, stage):
    """
    Parameters
    ----------
    input_tensor : Output of last layer
    alpha : Number of neuron
    stage : Index of hidden layer

    Returns output tensor
    -------
    """
    name_base = 'HL' + stage
    x = layers.Dense(alpha, name=name_base + '_1')(input_tensor)
    x = layers.BatchNormalization(name=name_base + '_2')(x)
    x = layers.Activation('relu', name=name_base + '_3')(x)
    x = layers.Dropout(0.2, name=name_base + '_4')(x)
    x = layers.add([x, input_tensor], name=name_base + '_Add')
    return x


def AUD(alpha, N):
    model_input = layers.Input(shape=[2*m, ], name='InputLayer')
    x = layers.Dense(alpha, name='InputFC')(model_input)
    x = layers.BatchNormalization(name='InputBN')(x)
    x = Hidden_Layer(x, alpha, stage='_A')
    x = Hidden_Layer(x, alpha, stage='_B')
    x = Hidden_Layer(x, alpha, stage='_C')
    x = Hidden_Layer(x, alpha, stage='_D')
    x = Hidden_Layer(x, alpha, stage='_E')
    x = Hidden_Layer(x, alpha, stage='_F')
    x = layers.Dense(N, name='OutputFC')(x)
    x = layers.Softmax(axis=-1, name='OutputActivation')(x)

    model = Model(model_input, x, name='D_AUD')
    return model



# %% Training data generation
p = 10**3
y_tilde, p_hat = train_data_gen(N, m, Nd, dv, p, k)
y_tilde = y_tilde.T*1/k
p_hat = p_hat.T
# y_hat, p = train_data_gen(N, m, Nd, dv, p, k, training_SNR)
# sio.savemat('p5.mat', {'p': p,'y_hat': y_hat})


# %% Training
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=7, min_lr=0.000001)

data = sio.loadmat('p1')
y_hat = data['y_hat']
p = data['p']
y_hat = y_hat.T
p = p.T/k
AUD1 = AUD(alpha, N)
AUD1.compile(optimizer=Adam(learning_rate=5*10**-4),
             loss='categorical_crossentropy')
AUD1.fit(y_hat, p,
         epochs=45,
         batch_size=1024,
         validation_split=0.3,
         callbacks=[early_stopping, reduce_lr])
AUD1.save_weights('./first_weight.h5')
hist_dict = AUD1.history
all_val_loss = hist_dict.history['val_loss']
all_loss = hist_dict.history['loss']

# model.summary()
# a=tf.keras.utils.plot_model(model)


# plot validation loss
epoch = np.arange(1, 54 + 1)

plt.semilogy(epoch, all_val_loss, label='val_loss')
plt.semilogy(epoch, all_loss, label='loss')

plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('Binary cross-entropy loss')


# plot P success
test = 10000
training_SNR = np.arange(4, 17, 2)
error_rate = np.zeros(training_SNR.shape,)
for i in range(len(training_SNR)):
    # print(i)
    y_test, p_test = train_data_gen(N, m, Nd, dv, test, k, training_SNR[i])
    y_test = y_test.T
    p_hat = AUD1.predict(y_test)
    p_hat = (p_hat > (1/k)).astype(int)
    z = np.where((p_hat.reshape(test*100,)-p_test.T.reshape(test*100,)) != 0)
    error_rate[i] = z[0].size/test/100
