import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
import multiprocessing
import socket
from collections.abc import Iterable
#from tensorflow.keras.backend import set_session, clear_session

import pfsspec.util as util
from pfsspec.notebookrunner import NotebookRunner

class Script():
    def __init__(self, logging=True):
        self.parser = None
        self.args = None
        self.debug = False
        self.random_seed = None
        self.logging = logging
        self.log_level = None
        self.logging_console_handler = None
        self.logging_file_handler = None
        self.dir_history = []
        self.outdir = None
        self.is_batch = 'SLURM_JOBID' in os.environ
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            self.threads = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            self.threads = multiprocessing.cpu_count()

    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.add_subparsers(self.parser)

    def add_subparsers(self, parser):
        # Default behavior doesn't user subparsers
        self.add_args(parser)

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.is_arg(name, args)

    def add_args(self, parser):
        parser.add_argument('--config', type=str, nargs='+', help='Load config from json file.')
        parser.add_argument('--debug', action='store_true', help='Run in debug mode\n')
        parser.add_argument('--log-level', type=str, default=None, help='Logging level\n')
        parser.add_argument('--random-seed', type=int, default=None, help='Set random seed\n')

    def get_configs(self, args):
        configs = []
        if 'config' in args and args['config'] is not None:
            if isinstance(args['config'], Iterable):
                filenames = list(args['config'])
            else:
                filenames = [args['config']]

            for filename in filenames:
                config = self.load_args_json(filename)
                configs.append(config)

        return configs

    def parse_args(self):
        if self.args is None:
            # - 1. parse command-line args with defaults enabled (already done above)
            self.args = self.parser.parse_args().__dict__
            configs = self.get_configs(self.args)
            if len(configs) > 0:
                # If a config file is used:
                # - 2. load config file, override all specified arguments
                for config in configs:
                    self.merge_args(config, override=True, recursive=True)

                # - 3. reparse command-line with defaults suppressed, apply overrides
                self.disable_parser_defaults(self.parser)
                command_args = self.parser.parse_args().__dict__
                self.merge_args(command_args, override=True, recursive=False)

            # Parse some special but generic arguments
            if 'debug' in self.args and self.args['debug']:
                self.debug = True
            if 'log_level' in self.args and self.args['log_level'] is not None:
                self.log_level = self.args['log_level']
            if 'random_seed' in self.args and self.args['random_seed'] is not None:
                self.random_seed = self.args['random_seed']

    def merge_args(self, other_args, override=True, recursive=False):
        if 'config' in other_args and recursive:
            # This is a config within a config file, load configs recursively, if requested
            configs = self.get_configs(other_args)
            for config in configs:
                self.merge_args(config, override=override, recursive=True)

        for k in other_args:
            if other_args[k] is not None and (k not in self.args or self.args[k] is None or override):
                self.args[k] = other_args[k]

    def disable_parser_defaults(self, parser):
        # Call recursively for subparsers
        for a in parser._actions:
            if isinstance(a, (argparse._StoreAction, argparse._StoreConstAction,
                              argparse._StoreTrueAction, argparse._StoreFalseAction)):
                a.default = None
            elif isinstance(a, argparse._SubParsersAction):
                for k in a.choices:
                    if isinstance(a.choices[k], argparse.ArgumentParser):
                        self.disable_parser_defaults(a.choices[k])

    @staticmethod
    def get_env_vars(prefix='PFSSPEC'):
        vars = {}
        for k in os.environ:
            if k.startswith(prefix):
                vars[k] = os.environ[k]
        return vars

    @staticmethod
    def substitute_env_vars(data, vars=None):
        vars = vars or Script.get_env_vars()

        if isinstance(data, dict):
            return {k: Script.substitute_env_vars(data[k], vars) for k in data}
        elif isinstance(data, list):
            return [Script.substitute_env_vars(d, vars) for d in data]
        elif isinstance(data, tuple):
            return tuple([Script.substitute_env_vars(d, vars) for d in data])
        elif isinstance(data, str):
            for k in vars:
                data = data.replace(vars[k], '${' + k + '}')
            return data
        else:
            return data

    @staticmethod
    def resolve_env_vars(data, vars=None):
        vars = vars or Script.get_env_vars()

        if isinstance(data, dict):
            return {k: Script.resolve_env_vars(data[k], vars) for k in data}
        elif isinstance(data, list):
            return [Script.resolve_env_vars(d, vars) for d in data]
        elif isinstance(data, tuple):
            return tuple([Script.resolve_env_vars(d, vars) for d in data])
        elif isinstance(data, str):
            for k in vars:
                data = data.replace('${' + k + '}', vars[k])
            return data
        else:
            return data

    @staticmethod
    def dump_json_default(obj):
        if isinstance(obj, float):
            return "%.5f" % obj
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
        args = Script.substitute_env_vars(self.args)
        with open(filename, 'w') as f:
            json.dump(args, f, default=Script.dump_json_default, indent=4)

    def dump_args_yaml(self, filename):
        args = Script.substitute_env_vars(self.args)
        with open(filename, 'w') as f:
            yaml.dump(args, f, indent=4)

    def load_args_json(self, filename):
        with open(filename, 'r') as f:
            args = json.load(f)
        args = Script.resolve_env_vars(args)
        return args

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
        elif self.log_level is not None:
            return getattr(logging, self.log_level)
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
    
    def init_logging(self, outdir):
        if self.logging:
            self.setup_logging(os.path.join(outdir, type(self).__name__.lower() + '.log'))
            self.save_command_line(os.path.join(outdir, 'command.sh'))
            self.dump_env(os.path.join(outdir, 'env.sh'))
            self.dump_args_json(os.path.join(outdir, 'args.json'))

    def execute(self):
        self.prepare()
        self.run()
        self.finish()

    def prepare(self):
        self.create_parser()
        self.parse_args()
        if self.logging:
            self.setup_logging()
        if self.debug:
            np.seterr(all='raise')
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

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