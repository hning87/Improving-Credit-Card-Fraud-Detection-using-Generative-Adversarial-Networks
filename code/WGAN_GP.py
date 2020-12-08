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
from keras.optimizers import RMSprop

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
batch_size=128

# %% ----------------------------------- GAN ---------------------------------------------------------------------------
# https://github.com/keras-team/keras-io/blob/master/examples/generative/wgan_gp.py

def generator_wgan_gp():
    noise = Input(shape=(latent_dim,))

    x = Dense(64, kernel_initializer=weight_init)(noise)
    x = LeakyReLU(0.2)(x)

    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(256, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    # tanh is removed since we are not dealing with normalized image data
    out = Dense(data_dim, kernel_initializer=weight_init)(x)

    model = Model(inputs=noise, outputs=out)

    return model


def discriminator_wgan_gp():
    data = Input(shape=data_dim)
    x = Dense(256, kernel_initializer=weight_init)(data)
    x = LeakyReLU(0.2)(x)

    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(64, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    out = Dense(1, kernel_initializer=weight_init)(x)
    model = Model(inputs=data, outputs=out)

    return model


# %% ----------------------------------- GAN ----------------------------------------------------------------------
# gp_lambda = 5 # change lambda for gradient penalty

class WGAN_GP(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=2,
        gp_lambda=5,
    ):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_lambda = gp_lambda

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_data, fake_data):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated data
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated data.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_data):
        if isinstance(real_data, tuple):
            real_data = real_data[0]

        # Get the batch size
        batch_size = tf.shape(real_data)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 1 extra steps
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_data = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_data, training=True)
                # Get the logits for real images
                real_logits = self.discriminator(real_data, training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_score=real_logits, fake_score=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_lambda

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake data using the generator
            generated_data = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake data
            gen_data_logits = self.discriminator(generated_data, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_data_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Optimizer for both the networks

# generator_optimizer = Adam(learning_rate=0.00001)
# discriminator_optimizer = Adam(learning_rate=0.00001)

generator_optimizer = Adam(learning_rate=0.00001, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = Adam(learning_rate=0.00001, beta_1=0.5, beta_2=0.9)

# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_score, fake_score):
    real_loss = tf.reduce_mean(real_score)
    fake_loss = tf.reduce_mean(fake_score)
    return fake_loss - real_loss


# Define the loss functions to be used for generator
def generator_loss(fake_score):
    return -tf.reduce_mean(fake_score)


d_model = discriminator_wgan_gp()
g_model = generator_wgan_gp()


# Get the wgan model
wgan_gp = WGAN_GP(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
    discriminator_extra_steps=2,
)

# Compile the wgan model
wgan_gp.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# load data
df = pd.read_csv('creditcard.csv')
data, df_fraud = utils.preprocessing(df)

x_train, x_test, y_train, y_test = utils.split(df_fraud)
X_train = x_train.to_numpy()

# Epochs to train
epochs = 1200
X_train_fraud = X_train.to_numpy()
wgan_gp.fit(X_train_fraud, batch_size=128, epochs=epochs)

wgan_gp.generator.save('wgan_gp_generator.h5')

