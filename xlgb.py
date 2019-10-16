# Wallethub 2019
# Luis Castro
#
# "Black-box" class that receives hyperparameters from the BO algorithm and 
# and trains ML models with KFold CV. Outputs a metric to be optimized. 
#
import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from utils import rmse, mae_thresh, prepare_params

class XL_GB(object):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, config: dict, params_xgb: dict, params_lgb: dict):
        """
        Initialization of the class, takes the features and target values as well as the general configuration
        and the configurations for both LGB and XGB ML algorithms

        Creates a feature importance dataset which will be filled incrementally with the gain information of the 
        algorithms when they run and later will be save to file.

        X: pd.DataFrame
            Features or data of the dataset
        
        y: pd.DataFrame
            Target, this is what is going to be predicted

        config: dict
            Overall configuration parameters
        
        params_xgb: dict
            Configuration parameters and hyperparameters for Xgboost
        
        params_lgb: dict
            Configuration parameters and hyperparameters for Light GBM
        """
        self.X = X
        self.y = y
        self.config = config
        self.params_xgb = params_xgb
        self.params_lgb = params_lgb
        self.count = 0
        self._feature_importance()


    def _feature_importance(self):
        """
        Helper method that initializes a feature importance dataset.
        """
        self.feature_importance = pd.DataFrame({'feature': self.X.columns})
        self.feature_importance['gain_xgb'] = 0.0
        self.feature_importance['gain_lgb'] = 0.0 


    def fit_predict(self, **kwargs) -> float:
        """
        The fit_predict method is created this way, as a black box for the Bayesian optimization
        library to work. It accepts the hyperparamters and creates the predictions with them.

        It keeps track of 3 metrics (mae, rmse and theshold mae) using a KFold crossvalidation and 
        one or both of the algorithms.

        This approach was used (instead of using cv included in the packages) for getting and combining
        predictions at the crossvalidation stage. Also, different kind of manipulations can be made
        with the data prior to prediction.

        kwargs: dict
            key / value pairs of the hyperparameters of xgb and/or lgb

        Returns
        -------
            float
                The selected metric to be optimized.
        """
        rmse_final = 0.0
        mae_final = 0.0
        mae_thresh_final = 0.0

        rmse_final_train = 0.0
        mae_final_train = 0.0
        mae_thresh_final_train = 0.0

        self.params_xgb = prepare_params(self.params_xgb, self.config, kwargs, 'xgb')
        self.params_lgb = prepare_params(self.params_lgb, self.config, kwargs, 'lgb')

        folds = KFold(n_splits=self.config['n_fold'], shuffle=self.config['shuffle'], random_state=self.config['random_state'])

        for train_index, valid_index in folds.split(self.X, self.y):
            X_train, X_valid = self.X.iloc[train_index], self.X.iloc[valid_index]
            y_train, y_valid = self.y.iloc[train_index], self.y.iloc[valid_index]

            if self.config['algorithm'] in ['xgb', 'both']:
                xgb_train = xgb.DMatrix(X_train, y_train)
                xgb_valid = xgb.DMatrix(X_valid, y_valid)

                bst_xgb = xgb.train(self.params_xgb,
                                    xgb_train,
                                    num_boost_round=self.config['num_boost_round'],
                                    early_stopping_rounds=self.config['early_stopping_rounds'],
                                    verbose_eval=self.config['verbose'],
                                    evals=[(xgb_valid, 'eval')])
                
                xgb_best_iteration = bst_xgb.best_iteration
                
                xgb_importance = bst_xgb.get_score(importance_type='total_gain')
                self.feature_importance['gain_xgb'] += self.feature_importance['feature'].apply(lambda x: xgb_importance[x] if x in xgb_importance else 0.0)
                
                xgb_valid_pred = bst_xgb.predict(xgb.DMatrix(X_valid), ntree_limit=xgb_best_iteration)
                xgb_train_pred = bst_xgb.predict(xgb.DMatrix(X_train), ntree_limit=xgb_best_iteration)


            if self.config['algorithm'] in ['lgb', 'both']: 
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_valid = lgb.Dataset(X_valid, y_valid)

                bst_lgb = lgb.train(self.params_lgb,
                                    lgb_train,
                                    num_boost_round=self.config['num_boost_round'],
                                    early_stopping_rounds=self.config['early_stopping_rounds'],
                                    verbose_eval=self.config['verbose_eval'],
                                    valid_sets=[lgb_valid])

                lgb_best_iteration = bst_lgb.best_iteration

                self.feature_importance['gain_lgb'] += np.array(bst_lgb.feature_importance(importance_type='gain'))
                
                lgb_valid_pred = bst_lgb.predict(X_valid, num_iteration=lgb_best_iteration)
                lgb_train_pred = bst_lgb.predict(X_train, num_iteration=lgb_best_iteration)

            if self.config['algorithm'] == 'xgb':
                y_pred = xgb_valid_pred
                y_pred_train = xgb_train_pred
            elif self.config['algorithm'] == 'lgb':
                y_pred = lgb_valid_pred
                y_pred_train = lgb_train_pred
            elif self.config['algorithm'] == 'both':
                balance = kwargs['balance']
                y_pred = balance * xgb_valid_pred + (1.0 - balance) * lgb_valid_pred
                y_pred_train = balance * xgb_train_pred + (1.0 - balance) * lgb_train_pred

            valid_mae_thresh = mae_thresh(y_valid, y_pred, thresh=self.config['threshold'])
            valid_rmse = rmse(y_valid, y_pred)
            valid_mae = mean_absolute_error(y_valid, y_pred)

            mae_thresh_final += valid_mae_thresh / self.config['n_fold']
            rmse_final += valid_rmse / self.config['n_fold']
            mae_final += valid_mae / self.config['n_fold']
            
            train_mae_thresh = mae_thresh(y_train, y_pred_train, thresh=self.config['threshold'])
            train_rmse = rmse(y_train, y_pred_train)
            train_mae = mean_absolute_error(y_train, y_pred_train)

            mae_thresh_final_train += train_mae_thresh / self.config['n_fold']
            rmse_final_train += train_rmse / self.config['n_fold']
            mae_final_train += train_mae / self.config['n_fold']

        self.count += 1

        print('Iteration {} - Validation / Train - MAE: {:.3f} / {:.3f}, RMSE: {:.3f} / {:.3f}, MAE above {}: {:.3f} / {:.3f}'.format(self.count, 
                                    mae_final, mae_final_train, rmse_final, rmse_final_train, self.config['threshold'], mae_thresh_final, mae_thresh_final_train))
        self.feature_importance.to_csv(self.config['feat_imp_path'], index=None)

        if self.config['bo_optimize'] == 'rmse': result = -rmse_final
        elif self.config['bo_optimize'] == 'mae': result = -mae_final
        elif self.config['bo_optimize'] == 'mae_t': result = -mae_thresh_final

        return result