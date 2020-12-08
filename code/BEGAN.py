import numpy as np
import os
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
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
weight_init = glorot_normal(seed=SEED)

# %% ---------------------------------- Hyperparameters ----------------------------------------------------------------
latent_dim = 32
data_dim = 30
n_classes = 2
optimizer = Adam(lr=0.0001, beta_1=0.1, beta_2=0.9)

# %% ----------------------------------- GAN ---------------------------------------------------------------------------
# Build Encoder
def encoder():
    data = Input(shape=(data_dim,))

    x = Dense(256, kernel_initializer=weight_init)(data)
    x = LeakyReLU(0.2)(x)

    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(64, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    encodered = Dense(latent_dim)(x)

    model = Model(inputs=data, outputs=encodered)
    return model


def decoder():
    noise = Input(shape=(latent_dim,))

    x = Dense(64, kernel_initializer=weight_init)(noise)
    x = LeakyReLU(0.2)(x)

    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(256, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    generated = Dense(data_dim, kernel_initializer=weight_init)(x)

    generator = Model(inputs=noise, outputs=generated)
    return generator


def generator(decoder):
    noise = Input(shape=(latent_dim,))

    generated_feature = decoder(noise)
    model = Model(inputs=noise, outputs=generated_feature)

    return model


def discriminator(encoder, decoder):
    data = Input(shape=(data_dim,))

    x = encoder(data)
    out = decoder(x)

    model = Model(inputs=data, outputs=out)
    model.compile(optimizer=optimizer, loss=l1Loss)
    return model


def l1Loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def train_G(generator, discriminator):
    # Freeze the discriminator when training generator
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=optimizer, loss=l1Loss)

    return model


class BEGAN:
    def __init__(self, g_model, d_model, kLambda=0.000001, k=0.06):
        self.z = latent_dim
        self.optimizer = optimizer

        self.generator = g_model
        self.discriminator = d_model
        self.kLambda = kLambda
        self.k = k

        self.train_G = train_G(self.generator, self.discriminator)
        self.loss_D, self.loss_G = [], []

    def train(self, data, batch_size, steps_per_epoch=100, gamma=0.5):

        for epoch in range(steps_per_epoch):
            # Select a random batch of transactions data
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            # generate a batch of new data
            noise_D = np.random.normal(0, 1, size=(batch_size, latent_dim))
            fake_data_D = self.generator.predict(noise_D)

            # Train D
            d_loss_real = self.discriminator.train_on_batch(real_data, real_data)

            weights = -self.k * np.ones(batch_size)
            d_loss_fake = self.discriminator.train_on_batch(fake_data_D, fake_data_D, weights)
            d_loss = d_loss_real + d_loss_fake
            self.loss_D.append(d_loss)

            # Train G
            noise_G = np.random.normal(0, 1, size=(batch_size, latent_dim))
            fake_data_G = self.generator.predict(noise_G)
            loss_g = self.train_G.train_on_batch(noise_G, fake_data_G)
            self.loss_G.append(loss_g)

            # Update k
            self.k = self.k + self.kLambda * (gamma * d_loss_real - loss_g)
            self.k = min(max(self.k, 1e-05), 1)

            # Report Results
            m_global = d_loss + np.abs(gamma * d_loss_real - loss_g)

            if (epoch + 1) * 10 % steps_per_epoch == 0:
                print('Steps (%d / %d): [loss_D: %f] [Loss_G: %f] [M_global: %f]' %
                      (epoch + 1, steps_per_epoch, 100 * self.loss_D[-1], self.loss_G[-1], m_global))

        return


D = discriminator(encoder(), decoder())
G = generator(decoder())
trainer = BEGAN(g_model=G, d_model=D)


df = pd.read_csv('creditcard.csv')
data, df_fraud = utils.preprocessing(df)

x_train, x_test, y_train, y_test = utils.split(df_fraud)

EPOCHS = 20
X_train = x_train.to_numpy()
for epoch in range(EPOCHS):
    print('EPOCH # ', epoch + 1, '-' * 50)
    trainer.train(X_train, batch_size=64, steps_per_epoch=100)
    if (epoch+1) % 1 == 0:
        trainer.generator.save('began_generator.h5')


