import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold as SKF
import math


def read_csv(path):
    return pd.read_csv(path)

def split_xy(df, label):
    return df.drop(label, axis=1), df[label]

def remove_null(X, y):
    nulls = y.isnull()
    return X[~nulls], y[~nulls]

def random_split(X, y, n_splits, shuffle, random_state):
    skf = SKF(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for train_index, test_index in skf.split(X, y):
        yield X.values[train_index], y.values[train_index], X.values[test_index], y.values[test_index]

def time_split(X, y, n_splits):    
    size = X.shape[0]
    split_size = size // (n_splits)
    
    for i in range(n_splits-1):
        yield X.values[i*split_size:(i+1)*split_size], y.values[i*split_size:(i+1)*split_size], X.values[(i+1)*split_size:(i+2)*split_size], y.values[(i+1)*split_size:(i+2)*split_size]         

def to_lgb(X, y):
    return lgb.Dataset(X, label=y)

def to_xgb(X, y):
    return xgb.DMatrix(X, y)

def lgb_score(y_true, y_score):
    return roc_auc_score(y_true, y_score)

def xgb_score(preds, dtrain):
    score = roc_auc_score(y_true=dtrain.get_label(), y_score=preds)
    return 'auc-{}'.format(score), score

def reduce_mem_usage(props):
	start_mem_usg = props.memory_usage().sum() / 1024**2
	print('Initial Memory Usage {} MB.'.format(start_mem_usg))

	for col in props.columns:
		if props[col].dtype != object:
			IsInt = False
			mx = props[col].max()
			mn = props[col].min()

			if not np.isfinite(props[col]).all(): 
				props[col].fillna(mn-1,inplace=True)  

			asint = props[col].fillna(0).astype(np.int64)
			result = (props[col] - asint)
			result = result.sum()
			if result > -0.01 and result < 0.01:
				IsInt = True

			if IsInt:
				if mn >= 0:
					if mx < 255:
						props[col] = props[col].astype(np.uint8)
					elif mx < 65535:
						props[col] = props[col].astype(np.uint16)
					elif mx < 4294967295:
						props[col] = props[col].astype(np.uint32)
					else:
						props[col] = props[col].astype(np.uint64)
				else:
					if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
						props[col] = props[col].astype(np.int8)
					elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
						props[col] = props[col].astype(np.int16)
					elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
						props[col] = props[col].astype(np.int32)
					elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
						props[col] = props[col].astype(np.int64)    

			else:
				props[col] = props[col].astype(np.float32)

	mem_usg = props.memory_usage().sum() / 1024**2 
	print('Final Memory Usage {} MB.'.format(mem_usg))
	return props

class targetEncoder(object):
    def __init__(self, df, col, label='isFraud'):
        self.col = col
        self.label = label
        self.values = df[col].values
        self.labels = df[label].values
        self.uniqValues = set(df[col].unique())
        self.encodedValues = {}
        self.encodedValues = {x:{'sum':0, 'count':0} for x in self.uniqValues}
        
    def encodeValues(self):
        for val, label in zip(self.values,self.labels):
            if isinstance(val,str) or not math.isnan(val):
                self.encodedValues[val]['sum'] += label
                self.encodedValues[val]['count'] += 1
            
        for key in self.encodedValues.keys():
            self.encodedValues[key]['mean'] = self.encodedValues[key]['sum']/max(self.encodedValues[key]['count'],1)
            
    def returnValue(self, key):
        if key in self.encodedValues:
            return self.encodedValues[key]['mean']
        return -1.0