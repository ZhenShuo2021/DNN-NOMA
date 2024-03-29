# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:03:37 2021

@author: Leo
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
# tf.compat.v1.disable_eager_execution()


m = 70
N = 100
Nd = 7
p = 1024
k = 4
dv = 10
alpha = 10*N
snr = 10  # training snr


def SpreadingCodeGen(N, m, Nd, dv):
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
    return index_list


def CodebookGen(N, m, Nd, dv, SC):
    Codebook = np.zeros([m, m*N*Nd], dtype=int)
    for i in range(N):
        for j in range(dv):
            for k in range(Nd):
                Codebook[SC[i][j], (SC[i][j] + k*m + m*Nd*i)] = 1
    return Codebook


# Generate spreading code for each user
SC = SpreadingCodeGen(N, m, Nd, dv)
Codebook = CodebookGen(N, m, Nd, dv, SC)

# Check if codebook is correct
for i in range(5):
    test = np.random.randint(0, 100, size=1)
    test = test[0]  # to int
    if (Codebook[SC[test][0], SC[test][0]+m*Nd*test] == 1) and (
            Codebook[SC[test][1], SC[test][1]+m*Nd*test] == 1):
        ans = "Correct"
    else:
        ans = "Failure"
    print("Codebook test: " + ans)


def TrainingDataGen(N, m, Nd, dv, p, k, snr, Codebook):
    while True:
        # row: user, col: Nd symbols with m channel gains
        active_index_matrix = np.zeros((k, p), dtype='int32')   # List
        # np array generated by active_index_matrix
        active_delta_matrix = np.zeros((N, p), dtype='int32')
        y_hat_p = np.zeros((2*m, p))
        N0 = 10**(-snr/10)
        # for epoch in range(p):
        for i in range(p):
            active_index = np.random.choice(N, k, replace=False)
            active_index_matrix[:, i] = active_index
            active_delta_matrix[:, i][active_index] = 1

        x = np.zeros((m*Nd*N, p), dtype='complex128')
        for i in range(p):
            # print(i)
            x_temp = np.zeros((N, m*Nd), dtype='complex128')
            for j in (active_index_matrix[:, i]):
                bits = np.random.randint(0, 2, size=[1, Nd])*2-1
                # bits = np.ones((1, 7))
                channel = np.random.randn(1, m) + 1j*np.random.randn(1, m)
                x_temp[j, :] = np.kron(bits, channel)
            x[:, i] = x_temp.reshape(m*Nd*N,)
        y_tilde = np.dot(Codebook, x)
        noise = 1*np.sqrt(N0/2) * (np.random.randn(*y_tilde.shape,) +
                                   1j*np.random.randn(*y_tilde.shape,))
        # noise = np.sqrt(N0/2) * np.zeros(y_tilde.shape,)    # use for check (no noise)
        y_tilde = y_tilde + noise

        y_hat_p[:m, :] = np.real(y_tilde)
        y_hat_p[m:, :] = np.imag(y_tilde)
        y_hat_p = y_hat_p.T
        active_delta_matrix = active_delta_matrix.T
        yield (y_hat_p, active_delta_matrix)
    # return y_hat_p, active_delta_matrix


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


early_stopping = EarlyStopping(monitor='val_loss', patience=20)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=7, min_lr=0.000001)

AUD1 = AUD(alpha, N)
AUD1.compile(optimizer=Adam(learning_rate=5*10**-4),
             # loss=tf.keras.losses.CategoricalCrossentropy)
             loss='categorical_crossentropy')
AUD1.summary()
tf.keras.utils.plot_model(AUD1, to_file="model.png", dpi=150)


AUD1.fit(TrainingDataGen(N, m, Nd, dv, p, k, snr, Codebook),  # 一次產生p筆資料
         steps_per_epoch=10,  # 一個epoch會跑training_gen七次
         epochs=100,
         # batch_size=1024,
         # validation_split=0.3,
         validation_data=TrainingDataGen(N, m, Nd, dv, p, k, snr, Codebook),
         validation_steps=3,
         callbacks=[early_stopping, reduce_lr])
# AUD1.save_weights('./sc10k4.h5')
AUD1.load_weights('./sc10k4.h5')
hist_dict = AUD1.history
all_val_loss = hist_dict.history['val_loss']
all_loss = hist_dict.history['loss']


epoch = np.arange(1, len(all_loss) + 1)

plt.semilogy(epoch, all_val_loss, label='val_loss')
plt.semilogy(epoch, all_loss, label='loss')
plt.legend(loc=0)
plt.grid('true')
plt.title('Loss (dv=10, k=4)')
plt.xlabel('epochs')
plt.ylabel('Binary cross-entropy loss')


test = 1000
training_SNR = np.arange(0, 21, 2)
error_rate = np.zeros(training_SNR.shape,)
for i in range(len(training_SNR)):
    print(training_SNR[i])
    TDG = TrainingDataGen(N, m, Nd, dv, test, k, training_SNR[i], Codebook)
    y_test, p_test = next(TDG)
    p_hat = AUD1.predict(y_test)
    p_hat = np.argsort(-p_hat)[:, :k]
    # p_hat = (p_hat>=(1/k)).astype(int)
    temp = np.zeros([test, N], dtype='int')
    for j in range(k):
        temp[np.arange(test), p_hat[:, j]] = 1

    z = np.where((temp.reshape(test*100,)-p_test.reshape(test*100,)) != 0)
    error_rate[i] = z[0].size/(test*N)

p_succ = 1 - error_rate
plt.title('Psucc (dv=10, k=2)')
plt.grid('true')
plt.xlabel('SNR')
plt.ylabel('Psucc')
plt.xlim(0, 20)
plt.ylim(0.1, 1)
plt.xticks(training_SNR, training_SNR)
plt.plot(training_SNR, p_succ, marker='o')