import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing
import socket
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from pfsspec.notebookrunner import NotebookRunner

class Script():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None
        self.debug = False
        self.random_seed = None
        self.logging_console_handler = None
        self.logging_file_handler = None
        self.dir_history = []
        self.outdir = None
        self.is_batch = 'SLURM_JOBID' in os.environ
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            self.threads = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            self.threads = multiprocessing.cpu_count()

    def add_subparsers(self, parser):
        # Default behavior doesn't user subparsers
        self.add_args(parser)

    def add_args(self, parser):
        parser.add_argument('--debug', action='store_true', help='Run in debug mode\n')
        parser.add_argument('--random-seed', type=int, default=None, help='Set random seed\n')

    def parse_args(self):
        if self.args is None:
            self.args = self.parser.parse_args().__dict__
            if 'debug' in self.args and self.args['debug']:
                self.debug = True
            if 'random_seed' in self.args and self.args['random_seed'] is not None:
                self.random_seed = self.args['random_seed']

    def dump_json_default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                if obj.size < 100:
                    return obj.tolist()
                else:
                    return "(not serialized)"
            else:
                return obj.item()
        return "(not serialized)"

    def dump_json(self, obj, filename):
        with open(filename, 'w') as f:
            if type(obj) is dict:
                json.dump(obj, f, default=Script.dump_json_default, indent=4)
            else:
                json.dump(obj.__dict__, f, default=Script.dump_json_default, indent=4)

    def dump_args_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.args, f, default=Script.dump_json_default, indent=4)

    def dump_env(self, filename):
        with open(filename, 'w') as f:
            for k in os.environ:
                f.write('{}="{}"\n'.format(k, os.environ[k]))

    def load_json(self, filename):
        with open(filename, 'r') as f:
            return json.load(f)

    def create_output_dir(self, dir, cont=False):
        logging.info('Output directory is {}'.format(dir))
        if cont:
            if os.path.exists(dir):
                logging.info('Found output directory.')
            else:
                raise Exception("Output directory doesn't exist, can't continue.")
        elif os.path.exists(dir):
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

    def get_logging_level(self):
        if self.debug:
            return logging.DEBUG
        else:
            return logging.INFO

    def setup_logging(self, logfile=None):
        root = logging.getLogger()
        root.setLevel(self.get_logging_level())

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if logfile is not None and self.logging_file_handler is None:
            self.logging_file_handler = logging.FileHandler(logfile)
            self.logging_file_handler.setLevel(self.get_logging_level())
            self.logging_file_handler.setFormatter(formatter)
            root.addHandler(self.logging_file_handler)

        if self.logging_console_handler is None:
            self.logging_console_handler = logging.StreamHandler(sys.stdout)
            self.logging_console_handler.setLevel(self.get_logging_level())
            self.logging_console_handler.setFormatter(formatter)
            root.addHandler(self.logging_console_handler)

        logging.info('Running script on {}'.format(socket.gethostname()))

    def suspend_logging(self):
        if self.logging_console_handler is not None:
            self.logging_console_handler.setLevel(logging.ERROR)

    def resume_logging(self):
        if self.logging_console_handler is not None:
            self.logging_console_handler.setLevel(self.get_logging_level())

    def save_command_line(self, filename):
        mode = 'a' if os.path.isfile(filename) else 'w'
        with open(filename, mode) as f:
            if mode == 'a':
                f.write('\n')
                f.write('\n')
            f.write(' '.join(sys.argv))

    def init_tensorflow(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = args.reserve_vram
        if 'gpus' in self.args and self.args['gpus'] is not None:
            config.gpu_options.visible_device_list = self.args['gpus']
        session = tf.Session(config=config)
        set_session(session)

    def release_tensorflow(self):
        K.clear_session()

    def init_logging(self, outdir):
        self.setup_logging(os.path.join(outdir, type(self).__name__ + '.log'))
        self.save_command_line(os.path.join(outdir, 'command.sh'))
        self.dump_env(os.path.join(outdir, 'env.sh'))
        self.dump_args_json(os.path.join(outdir, 'args.json'))

    def execute(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.prepare()
        self.run()
        self.finish()

    def prepare(self):
        self.add_subparsers(self.parser)
        self.parse_args()
        self.setup_logging()
        if self.debug:
            np.seterr(all='raise')

    def run(self):
        raise NotImplementedError()

    def finish(self):
        self.execute_notebooks()

    def execute_notebooks(self):
        pass

    def execute_notebook(self, notebook_name, output_notebook_name=None, output_html=True, parameters={}, kernel='python3', outdir=None):
        # Note that jupyter kernels in the current env might be different from the ones
        # in the jupyterhub environment

        logging.info('Executing notebook {}'.format(notebook_name))

        if outdir is None:
            outdir = self.args['out']

        # Project path is added so that the pfsspec lib can be called without
        # installing it
        if 'PROJECT_PATH' not in parameters:
            parameters['PROJECT_PATH'] = os.getcwd()

        if output_notebook_name is None:
            output_notebook_name = notebook_name

        nr = NotebookRunner()
        nr.input_notebook = os.path.join('nb', notebook_name + '.ipynb')
        nr.output_notebook = os.path.join(outdir, output_notebook_name + '.ipynb')
        if output_html:
            nr.output_html = os.path.join(outdir, output_notebook_name + '.html')
        nr.parameters = parameters
        nr.kernel = kernel
        nr.run()