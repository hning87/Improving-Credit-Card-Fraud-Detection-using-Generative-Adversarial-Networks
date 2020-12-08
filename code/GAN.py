import numpy as np
import os
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
import random
from keras.optimizers import Adam
import utils
import pandas as pd


# %% --------------------------------------- Set Seeds -----------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)

# %% ---------------------------------- Hyperparameters ----------------------------------------------------------------
latent_dim = 32
data_dim = 30
n_classes = 2
optimizer = Adam(lr=0.00001)

# %% ----------------------------------- GAN ---------------------------------------------------------------------------
def generator():
    noise = Input(shape=(latent_dim,))

    x = Dense(64, kernel_initializer=weight_init)(noise)
    #     x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(128, kernel_initializer=weight_init)(x)
    #     x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(256, kernel_initializer=weight_init)(x)
    #     x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # tanh is removed since we are not dealing with normalized image data
    out = Dense(data_dim, kernel_initializer=weight_init)(x)

    model = Model(inputs=noise, outputs=out)

    return model


def discriminator():
    data = Input(shape=data_dim)
    x = Dense(256, kernel_initializer=weight_init)(data)
    x = LeakyReLU(0.2)(x)
    #    x = layers.Dropout(0.1)(x)

    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    #    x = layers.Dropout(0.1)(x)

    x = Dense(64, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    #    x = layers.Dropout(0.1)(x)

    out = Dense(1, activation='sigmoid', kernel_initializer=weight_init)(x)

    model = Model(inputs=data, outputs=out)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_G(generator, discriminator):
    # Freeze the discriminator when training generator
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model


# %% ----------------------------------- GAN ----------------------------------------------------------------------
# modified from https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

class GAN:
    def __init__(self, g_model, d_model):
        self.z = latent_dim
        self.optimizer = optimizer

        self.generator = g_model
        self.discriminator = d_model

        self.train_G = train_G(self.generator, self.discriminator)
        self.loss_D, self.loss_G = [], []

    def train(self, data, batch_size=128, steps_per_epoch=50):

        for epoch in range(steps_per_epoch):
            # Select a random batch of transactions data
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            # generate a batch of new data
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            fake_data= self.generator.predict(noise)

            # Train D
            loss_real = self.discriminator.train_on_batch(real_data, np.ones(batch_size))
            loss_fake = self.discriminator.train_on_batch(fake_data, np.zeros(batch_size))
            self.loss_D.append(0.5 * np.add(loss_fake, loss_real))

            # Train G
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            loss_G = self.train_G.train_on_batch(noise, np.ones(batch_size))
            self.loss_G.append(loss_G)

            if (epoch + 1) * 10 % steps_per_epoch == 0:
                print('Steps (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc: %.2f%%] [Loss_G: %f]' %
                  (epoch+1, steps_per_epoch, loss_real[0], loss_fake[0], 100*self.loss_D[-1][1], loss_G))

        return


D = discriminator()
G = generator()

D.summary()
G.summary()

df = pd.read_csv('creditcard.csv')
data, df_fraud = utils.preprocessing(df)

x_train, x_test, y_train, y_test = utils.split(df_fraud)

EPOCHS = 10
X_train = x_train.to_numpy()
gan = GAN(g_model=G, d_model=D)

for epoch in range(EPOCHS):
    print('EPOCH # ', epoch + 1, '-' * 50)
    gan.train(X_train, batch_size=128, steps_per_epoch=100)
    if (epoch+1) % 1 == 0:
        gan.generator.save('gan_generator.h5')

