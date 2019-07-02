import os
import sys
import logging
import json
import numpy as np

logging_console_handler = None
logging_file_handler = None
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

def setup_logging(logfile=None):
    global logging_console_handler
    global logging_file_handler

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if logfile is not None and logging_file_handler is None:
        logging_file_handler = logging.FileHandler(logfile)
        logging_file_handler.setLevel(logging.INFO)
        logging_file_handler.setFormatter(formatter)
        root.addHandler(logging_file_handler)

    if logging_console_handler is None:
        logging_console_handler = logging.StreamHandler(sys.stdout)
        logging_console_handler.setLevel(logging.INFO)
        logging_console_handler.setFormatter(formatter)
        root.addHandler(logging_console_handler)

def parse_labels_coeffs(args):
    labels = []
    coeffs = []
    for k in args.labels:
        if '/' in k:
            parts = k.split('/')
            labels.append(parts[0])
            coeffs.append(float(parts[1]))
        else:
            labels.append(k)
            coeffs.append(1.0)
    return labels, np.array(coeffs)
