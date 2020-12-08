import numpy as np
import os
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Embedding, multiply, Flatten
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
optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
trainRatio = 5

# %% ---------------------------------------- load data ----------------------------------------------------------------
df = pd.read_csv('creditcard.csv')
data, df_fraud = utils.preprocessing(df)

x_train, x_test, y_train, y_test = utils.split(df_fraud)

# %% ----------------------------------- GAN ---------------------------------------------------------------------------
# Build Encoder
def encoder():
    data = Input(shape=(data_dim,))
    x = Dense(256, kernel_initializer=weight_init)(data)
    x = LeakyReLU(0.2)(x)
    #    x = layers.Dropout(0.1)(x)

    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    #    x = layers.Dropout(0.1)(x)

    x = Dense(64, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    #    x = layers.Dropout(0.1)(x)

    encodered = Dense(latent_dim)(x)


    model = Model(inputs=data, outputs=encodered)
    return model

def decoder():
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
    generated = Dense(data_dim, kernel_initializer=weight_init)(x)

    generator = Model(inputs=noise, outputs=generated)
    return generator

def embedding_labeled_latent():
    label = Input((1,), dtype='int32')
    noise = Input((latent_dim,))

    le = Flatten()(Embedding(n_classes, latent_dim)(label))

    noise_le = multiply([noise, le])

    model = Model([noise, label], noise_le)

    return model

# Build Autoencoder
def autoencoder_trainer(encoder, decoder, embedding):

    label = Input((1,), dtype='int32')
    feature = Input(shape=(data_dim,))

    latent = encoder(feature)
    labeled_latent = embedding([latent, label])
    rec_feature = decoder(labeled_latent)
    model = Model([feature, label], rec_feature)

    model.compile(optimizer=optimizer, loss='mae')
    return model

# Train Autoencoder
en = encoder()
de = decoder()
em = embedding_labeled_latent()
ae = autoencoder_trainer(en, de, em)

ae.fit([x_train, y_train], x_train,
       epochs=30,
       batch_size=128,
       shuffle=True,
       validation_data=([x_test, y_test], x_test))

# Build Discriminator loss
def discriminator_loss(real_logits, fake_logits, wrong_label_logits):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    wrong_label_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_label_logits, labels=tf.zeros_like(fake_logits)))

    return wrong_label_loss + fake_loss + real_loss


# Build generator loss
def generator_loss(fake_logits):
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return fake_loss


# Build Discriminator
def discriminator(encoder):
    label = Input((1,), dtype='int32')
    data = Input(shape=(data_dim,))

    x = encoder(data)

    le = Flatten()(Embedding(n_classes, 32)(label))
    le = Dense(32)(le)

    x_y = multiply([x, le])
    x_y = Dense(16)(x_y)

    out = Dense(1)(x_y)

    model = Model(inputs=[data, label], outputs=out)
    #     model.compile(optimizer=discriminator_optimizer, loss=discriminator_loss)

    return model

# Build Generator
def generator(decoder, embedding):
    noise = Input(shape=(latent_dim,))
    label = Input((1,), dtype='int32')

    labeled_latent = embedding([noise, label])
    generated_data = decoder(noise)
    model = Model(inputs=[noise, label], outputs=generated_data)

    return model

# %% --------------------------------------- BAGAN ---------------------------------------------------------------------
class BAGAN(Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
    ):
        super(BAGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(BAGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        if isinstance(data, tuple):
            real_data = data[0]
            labels = data[1]

        # Get the batch size
        batch_size = tf.shape(real_data)[0]

        # Get the latent vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
        wrong_labels = tf.random.uniform((batch_size,), 0, n_classes)
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator([random_latent_vectors, fake_labels], training=True)
            # Get the logits for the fake images
            fake_logits = self.discriminator([fake_images, fake_labels], training=True)
            # Get the logits for real images
            real_logits = self.discriminator([real_data, labels], training=True)
            # Get the logits for wrong label classification
            wrong_label_logits = self.discriminator([real_data, wrong_labels], training=True)

            # Calculate discriminator loss using fake and real logits
            d_loss = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits,
                                    wrong_label_logits=wrong_label_logits
                                    )


        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, fake_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, fake_labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        return {"d_loss": d_loss, "g_loss": g_loss}


D = discriminator(en)
G = generator(de, em)

bagan = BAGAN(
    discriminator=D,
    generator=G,
    latent_dim=latent_dim,
)

generator_optimizer = Adam(learning_rate=0.001, beta_1=0.2, beta_2=0.9)

discriminator_optimizer = Adam(learning_rate=0.1, beta_1=0.2, beta_2=0.9)

# Compile the model
bagan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)

EPOCHS = 10
X_train = x_train.to_numpy()
Y_train = y_train.to_numpy()
for epoch in range(EPOCHS):
    print('EPOCHS # ', epoch + 1, '-' * 50)
    bagan.fit(X_train, Y_train, batch_size=258, epochs=1)

    if (epoch+1) % 1 == 0:
        bagan.generator.save('bagan_generator.h5')
