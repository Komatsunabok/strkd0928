import torch
import torch.nn as nn
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model_dict

def load_model(model_path, model_name, n_cls, gpu=None, opt=None):
    print('==> loading model')
    # model_t = get_teacher_name(model_path)
    model = model_dict[model_name](num_classes=n_cls)

    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt and opt.multiprocessing_distributed else 0)}
    
    state = torch.load(model_path, map_location=map_location)
    
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)

    print('==> done')
    return model

# model stateから'model'をとってくる
# 'model'しかないこともある
# <class 'dict'>
# dict_keys(['epoch', 'best_acc', 'model'])
# epoch: <class 'int'>
# best_acc: <class 'float'>
# model: <class 'collections.OrderedDict'>