import os
import logging
import json
import numpy as np

dir_history = []

def create_output_dir(dir):
    logging.info('Output directory is {}'.format(dir))
    if os.path.exists(dir):
        if len(os.listdir(dir)) != 0:
            raise Exception('Output directory is not empty.')
    else:
        logging.info('Creating output directory {}'.format(dir))
        os.makedirs(dir)

def dump_json_default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            if obj.size < 100:
                return obj.tolist()
            else:
                return "(not serialized)"
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def dump_json(args, filename):
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, default=dump_json_default, indent=4)

def pushd(dir):
    dir_history.append(os.getcwd())
    os.chdir(dir)

def popd():
    os.chdir(dir_history[-1])
    del dir_history[-1]