from imblearn.over_sampling import RandomOverSampler
from xgboost.sklearn import XGBClassifier


def XGBC_model_predit(x_train, y_train, x_test):
    ros = RandomOverSampler()
    x, y = ros.fit_sample(x_train, y_train)
    clf_xgb_os = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                               colsample_bynode=1, colsample_bytree=1, eval_metric='auc',
                               gamma=0.2, gpu_id=0, importance_type='gain',
                               interaction_constraints='', learning_rate=0.02, max_delta_step=0,
                               max_depth=10, min_child_weight=2,
                               monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
                               n_estimators=800, n_jobs=-1, num_parallel_tree=1, random_state=42,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.7,
                               tree_method='gpu_hist', validate_parameters=1)

    clf_xgb_os.fit(x, y)
    y_pred = clf_xgb_os.predict(x_test.to_numpy())
    return y_pred