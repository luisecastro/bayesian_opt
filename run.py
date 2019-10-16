import sys
import yaml
from bayes_opt import BayesianOptimization as BO
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from xgb_bo import XGB_BO
from lgb_bo import LGB_BO


if __name__ == '__main__':
     alg_type = sys.argv[1]
     config_file = sys.argv[2]

    with open(config_file,'r') as stream:
        conf = yaml.load(stream, Loader=yaml.FullLoader)

    print('- Configuration read successfully -\n')

    if alg_type == 'xgb':
        clf = XGB_BO(c)

    elif alg_type == 'lgb':
