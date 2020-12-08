import pandas as pd
import tensorflow as tf
import utils
import classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# load data
df = pd.read_csv('creditcard.csv')

# data preprocessing
data, df_fraud = utils.preprocessing(df)

# data split
x_train, x_test, y_train, y_test = utils.split(df)

ros = RandomOverSampler()
rus = RandomUnderSampler()
x_train_os, y_train_os = ros.fit_sample(x_train, y_train)
x_train_us, y_train_us = rus.fit_sample(x_train, y_train)

print(y_train_os.value_counts())
print(y_train_us.value_counts())

# best params after gridsearch
clf_xgb_us = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='auc',
              gamma=0.2, gpu_id=0, importance_type='gain',
              interaction_constraints='', learning_rate=0.01, max_delta_step=0,
              max_depth=6, min_child_weight=2,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=600, n_jobs=-1, num_parallel_tree=1, random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.7,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)

clf_xgb_us.fit(x_train_us, y_train_us)
y_pred_us = clf_xgb_us.predict(x_test)

utils.check_performance(y_test, y_pred_us)
utils.plot_cm(y_test, y_pred_us, 'RUS')

# best params after gridsearch
clf_xgb_os = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='auc',
              gamma=0.2, gpu_id=0, importance_type='gain',
              interaction_constraints='', learning_rate=0.02, max_delta_step=0,
              max_depth=10, min_child_weight=2,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=800, n_jobs=-1, num_parallel_tree=1, random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.7,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)
clf_xgb_os.fit(x_train_os, y_train_os)


clf_xgb_os.fit(x_train_os, y_train_os)
y_pred_os = clf_xgb_os.predict(x_test)

utils.check_performance(y_test, y_pred_os)
utils.plot_cm(y_test, y_pred_os, 'ROS')