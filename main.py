# Wallethub 2019
# Luis Castro
#
# Main function to train the model.
#

import numpy as np
import pandas as pd

from datetime import datetime
import sys
import yaml

from bayes_opt import BayesianOptimization as BO
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from utils import reduce_mem_usage
from xlgb import XL_GB

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    """
    To run:
        - python main.py config.yaml
    
    Please edit config.yaml file to suit your needs.
    """

    # Reads argument (path to the YAML configuration file)
    config_path = sys.argv[1]

    # Loads YAML file into memory
    with open(config_path, 'r') as stream:
        full_config = yaml.full_load(stream)

    # Assing parts of the configuration to variables
    # for readability
    config = full_config['config']
    params_xgb = full_config['params_xgb']
    params_lgb = full_config['params_lgb']
    bounds = full_config['bounds']
    bayesian = full_config['bayesian']

    print('- Configuration read successfully -\n')

    # Read dataset and split into features / target (X, y)
    # Reduce memory footprint if posible
    df = pd.read_csv(config['file_path'])
    df = reduce_mem_usage(df)

    X = df.drop(config['target'], axis=1)
    y = df[config['target']]
    del(df)

    print('- Dataset read successfully -\n')

    # Create an instance of the XL_GB class to be run by the 
    # Bayessian optimization algorithm
    reg = XL_GB(X, y, config, params_xgb, params_lgb)

    optimizer = BO(f=reg.fit_predict,
                    pbounds=bounds,
                    random_state=config['random_state'],
                    verbose=config['verbose'])

    # Load information from previous runs if available
    if bayesian['run_l']:
        load_logs(optimizer, logs=bayesian['log_in'])
        print('Optimizer loaded {} points from logs.\n'.format(len(optimizer.space)))

    # Asign unique name to the log files to be generated
    log_DT = datetime.now().timetuple()
    logger = JSONLogger(path=bayesian['log_out']+"/xl_gb_{}_{}_{}_{}_{}.json".format(log_DT[0], log_DT[1], log_DT[2], log_DT[3], log_DT[4]))

    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # Run the optimizer
    optimizer.maximize(init_points=bayesian['bo_seeds'],
                        n_iter=bayesian['bo_iter'],
                        acq=bayesian['acq'])

    # When finished, print the best iteration score and parameters
    print(optimizer.max['target'])
    print(optimizer.max['params'])