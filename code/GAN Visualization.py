import pandas as pd
import tensorflow as tf
import utils

# load data
df = pd.read_csv('creditcard.csv')

# data preprocessing
data, df_fraud = utils.preprocessing(df)
fraud = df_fraud.loc[:, df.columns != 'Class']

# load generator
generator = tf.keras.models.load_model('gan_pre_generator.h5')

# generate fraud data
gen_1000 = utils.gen_data(generator, 1000)
df_gen = pd.DataFrame(data=gen_1000, index=None, columns=fraud.columns)
print(df_gen.describe())

# compare with real data
utils.boxplot_compare(fraud, df_gen, 'green', 'purple', 'GAN with Pre-Train',
                      'Original Data Distribution versus GAN with Pre-Train')
