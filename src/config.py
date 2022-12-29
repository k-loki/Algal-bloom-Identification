"""Configuration file for each run."""


import time

# dot dictionary
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Config
config = {}

config['RANDOM_SEED'] = 18952

config['PROJECT_NAME'] = 'tick-tick-bloom'
config['DATA_DIR'] = '../data/'
config['MODEL_DIR'] = '../models/'

config['IMG_SIZE'] = (136, 136)

config['name'] = 'test run'   # 'conv2d_32_64_img_1k'

config['train'] =  dotdict({
        'epochs': 10,
        'batch_size': 32,
        'validation_split': 0.2,
        'shuffle': True,
        'verbose': 1
        })


config['unique_id'] = int(time.time())



config = dotdict(config)
