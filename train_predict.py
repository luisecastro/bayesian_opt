# Wallethub 2019
# Luis Castro
#
# Fit, save and predict with models 
#
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import lightgbm as lgb
import xgboost as xgb
import time
from utils import *
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)



if __name__ == '__main__':
    test_path = sys.argv[1]
    gpu_cpu = sys.argv[2]
    target = 'y'
    train = False
    predict = True

    params_xgb = {'booster': 'gbtree',
                    'silent': True,
                    'verbose': 0,
                    'tree_method': 'gpu_hist', 
                    'process_type': 'default',
                    'predictor': 'gpu_predictor',
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'random_state': 0,
                    'eta': 0.1,
                    'reg_alpha': 0.828526,
                    'reg_lambda': 1.384944,
                    'max_depth': 6,
                    'colsample_bylevel': 0.869632,
                    'colsample_bynode': 0.519594,
                    'colsample_bytree': 0.641403}

    params_lgb = {'objective': 'regression',
                    'save_binary': True,
                    'random_state': 0,
                    'feature_fraction_seed': 0,
                    'bagging_seed': 0,
                    'drop_seed': 0,
                    'data_random_seed': 0,
                    'boosting_type': 'gbdt',
                    'verbose': 0,
                    'metric': 'rmse',
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'reg_alpha': 0.635966,
                    'reg_lambda': 0.128295,
                    'max_depth': 7,
                    'feature_fraction': 0.560098,
                    'bagging_fraction': 0.988381}

    if train:
        train_path = 'X_rs.csv.gz'

        df = pd.read_csv(train_path)
        X = df.drop(target, axis=1)
        y = df[target]
        del(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.9, random_state=0)
  
        xgb_train = xgb.DMatrix(X_train, y_train)
        xgb_test = xgb.DMatrix(X_test, y_test)

        bst_xgb = xgb.train(params_xgb, xgb_train, num_boost_round=2048, early_stopping_rounds=32, verbose_eval=0, evals=[(xgb_test, 'eval')])
        bst_xgb.save_model('xgb_gpu.model')

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test)

        bst_lgb = lgb.train(params_lgb, lgb_train, num_boost_round=2048, early_stopping_rounds=32, verbose_eval=0, valid_sets=[lgb_test])
        bst_lgb.save_model('lgb_gpu.model')


    if predict:
        balance = 0.594361
        threshold = 3
        scaler_filename = 'scaler.save'

        df = pd.read_csv(test_path)
        
        tic = time.time()
        X = df.drop(target, axis=1)
        y = df[target]
        del(df)

        rs = joblib.load(scaler_filename) 

        X = rs.transform(X)

        if gpu_cpu == 'cpu':
            bst_lgb = lgb.Booster(model_file='lgb_cpu.model')
            bst_xgb = xgb.Booster(model_file='xgb_cpu.model')
        else:
            bst_lgb = lgb.Booster(model_file='lgb_gpu.model')
            bst_xgb = xgb.Booster(model_file='xgb_gpu.model')

        lgb_pred = bst_lgb.predict(X)
        xgb_pred = bst_xgb.predict(xgb.DMatrix(X))

        both_pred = balance * xgb_pred + (1.0 - balance) * lgb_pred

        mae_result = mean_absolute_error(y, both_pred)
        mae_t_result = mae_thresh(y, both_pred, threshold)
        rmse_result = rmse(y, both_pred)
        tac = time.time() - tic
        results = pd.DataFrame({'Train Elements':100000,'Test Elements':X.shape[0],'MAE':mae_result, 'MAE_3': mae_t_result, 
                                'RMSE':rmse_result, 'Time': tac, 'xgb_config':[params_xgb], 'lgb_config': [params_lgb]})
        
        results.T[0].to_csv('results.csv')
