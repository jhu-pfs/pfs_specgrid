import os
import sys
import json
import logging
import argparse
import numpy as np

class Script():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None
        self.logging_console_handler = None
        self.logging_file_handler = None
        self.dir_history = []

    def parse_args(self):
        self.args = self.parser.parse_args()

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

    def dump_json(self, obj, filename):
        with open(filename, 'w') as f:
            json.dump(obj.__dict__, f, default=Script.dump_json_default, indent=4)

    def dump_args_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.args.__dict__, f, default=Script.dump_json_default, indent=4)

    def create_output_dir(self, dir):
        logging.info('Output directory is {}'.format(dir))
        if os.path.exists(dir):
            if len(os.listdir(dir)) != 0:
                raise Exception('Output directory is not empty.')
        else:
            logging.info('Creating output directory {}'.format(dir))
            os.makedirs(dir)

    def pushd(self, dir):
        self.dir_history.append(os.getcwd())
        os.chdir(dir)

    def popd(self):
        os.chdir(self.dir_history[-1])
        del self.dir_history[-1]

    def setup_logging(self, logfile=None):
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if logfile is not None and self.logging_file_handler is None:
            self.logging_file_handler = logging.FileHandler(logfile)
            self.logging_file_handler.setLevel(logging.INFO)
            self.logging_file_handler.setFormatter(formatter)
            root.addHandler(self.logging_file_handler)

        if self.logging_console_handler is None:
            self.logging_console_handler = logging.StreamHandler(sys.stdout)
            self.logging_console_handler.setLevel(logging.INFO)
            self.logging_console_handler.setFormatter(formatter)
            root.addHandler(self.logging_console_handler)

    def add_args(self):
        self.parser.add_argument('--debug', action='store_true', help='Run in debug mode\n')

    def execute(self):
        self.prepare()
        self.run()

    def prepare(self):
        self.setup_logging()
        self.add_args()
        self.parse_args()
        if self.args.debug:
            np.seterr(all='raise')

    def run(self):
        self.setup_logging(os.path.join(self.outdir, 'training.log'))
        self.dump_args_json(os.path.join(self.outdir, 'args.json'))