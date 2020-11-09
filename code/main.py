import pandas as pd
import tensorflow as tf
import utils
import classification

# load data
df = pd.read_csv('creditcard.csv')
# data preprocessing
data, df_fraud = utils.preprocessing(df)
# load generator
generator = tf.keras.models.load_model('gan_pre_generator.h5')

# data split
x_train, x_test, y_train, y_test = utils.split(df)

# generate fraud data
gen_1000 = utils.gen_data(generator, 1000)
x_train_gen, y_train_gen = utils.concatenate(x_train, y_train, gen_1000)

# training classification
y_pred = classification.XGBC_model_predit(x_train_gen, y_train_gen, x_test)

# performance
utils.check_performance(y_test, y_pred)
utils.plot_cm(y_test, y_pred, 'GAN with Pre-Train')
