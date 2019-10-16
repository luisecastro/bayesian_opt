# Wallethub 2019
# Luis Castro
#
# Collection of auxiliary functions
#

import numpy as np
import pandas as pd
from sklearn.metrics import  mean_squared_error

def rmse(y_true: np.array, y_pred: np.array) -> float:
    """
    Custom metric that returns the root mean squared error of a couple
    of sets if predictions (because sklearn only has a mean squared error function)

    y_true: np.array
        Ground truth, true values

    y_pred: np.array
        Predictions from the ML algorith

    Returns
    -------
    float
        Root mean squared error of the predictions vs the true values
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_thresh(y_true: np.array, y_pred: np.array, thresh: float) -> float:
    """
    Custom metric that returns the average number of predictions whose mae
    is more than threshold away from the true value.

    y_true: np.array
        Ground truth, true values

    y_pred: np.array
        Predictions from the ML algorith

    threshold: float
        Max size of the absolute error to be considered a correct prediction

    Returns
    -------
    float
        Mean number of predictions whose mae was above the threshold

    """
    return (np.abs(y_true-y_pred)>thresh).mean()


def prepare_params(params: dict, config: dict, kwargs: dict, alg: str) -> dict:
    """
    Function that coerces value type to be the accepted types by the ML algorithms.

    params: dict
        parameters to be prepared for the BO algorithm

    config: dict
        variable that contains how to prepare the parameters
        list of which parameters are changed to which types
    
    kwargs: dict
        Selects the parameters to be included
    
    alg: str
        Select the algorithm for which to check the parameters (xgb, lgb)

    Returns
    -------
    dict
        params dictionary after coercing the values to specific types
    """
    for key, value in kwargs.items():
        right_key = key[:-4] if alg in key else key

        if key in params:
            if key in config['ints']:
                params[right_key] = int(value)
            elif key in config['floats']:
                params[right_key] = float(value)
            elif key in config['strs']:
                params[right_key] = str(value) 

    return params


def reduce_mem_usage(props: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory footprint from a dataset, blantantly stolen from Kaggle.
    Basically, it checks if the float values could be reduced to integers and 
    assigns the smallest size integer possible.
    
    props: pd.DataFrame
        Dataframe to be reduced in size
    
    Returns
    -------
    pd.DataFrame
        Dataframe after size reduction

    """
    start_mem_usg = round(props.memory_usage().sum() / 1024**2,2)
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

    mem_usg = round(props.memory_usage().sum() / 1024**2,2) 
    print('Final Memory Usage {} MB.'.format(mem_usg))
    return props


# Two sample tests, ideally most (all) functions would be tested thoroughly.
def test_rmse():
    assert(round(rmse(np.array([1,2,3,4]), np.array([4,3,2,1])), 2)==2.24)
    assert(round(rmse(np.array([0]), np.array([0])),2)==0.0)

def test_mae_thresh():
    assert(round(mae_thresh(np.array([1,2,3,4]), np.array([4,3,2,1]), 1.5), 2)==0.5)
    assert(round(mae_thresh(np.array([0]), np.array([0]), 1.5), 2)==0.0)