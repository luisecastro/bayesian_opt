import xgboost as xgb
from utils import *


class XGB_BO(object):
    def __init__(self, conf, param):
        self.conf = conf
        self.param = param
        self.count = 1
        self._start()
        
    def _start(self):
        df = reduce_mem_usage(read_csv(self.conf['path']))
        X, y = split_xy(df, self.conf['label'])
        X, y = remove_null(X, y)
        self.X = X
        self.y = y
        del([df, X, y])

    def fit_predict(self, **kwargs):
        result = 0.0
        conf = self.conf
        param = self.param
        best_iter = []
        
        for key, value in kwargs.items():
            if key in conf['ints']:
                param[key] = int(value)
            elif key in conf['floats']:
                param[key] = float(value)
            elif key in conf['strs']:
                param[key] = str(value)
        
        rs = random_split(self.X, self.y, conf['n_splits'], conf['shuffle'], conf['random_state'])
        
        for _ in range(conf['n_splits']):
            X_train, y_train, X_test, y_test = next(rs)
            xgb_train = to_xgb(X_train, y_train)
            xgb_test = to_xgb(X_test, y_test)
    
            bst = xgb.train(
                    params=param, 
                    dtrain=xgb_train, 
                    num_boost_round=conf['num_boost_round'], 
                    evals=[(xgb_test,'eval')], 
                    feval=xgb_score,
                    verbose_eval=conf['verbose_eval'], 
                    early_stopping_rounds=conf['early_stopping_rounds'], 
                    maximize=conf['maximize'])
            
            best_iteration = bst.best_ntree_limit
            
            y_pred = bst.predict(xgb_test, ntree_limit=best_iteration)
            
            result += roc_auc_score(y_test, y_pred) / conf['n_splits']
            
            best_iter.append(best_iteration)
            
    
        print('iteration-{} | test-score-{}\n'.format(self.count, result))
        print(best_iter)
        self.count += 1
        
        return result