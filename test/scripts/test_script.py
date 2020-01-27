import os

from test.test_base import TestBase
from pfsspec.scripts.script import Script

class TestScript(TestBase):
    def test_substitute_env_vars(self):
        data = 1
        data = Script.substitute_env_vars(data)
        self.assertEqual(1, data)

        data = [1, 2]
        data = Script.substitute_env_vars(data)
        self.assertEqual([1, 2], data)

        data = 'string'
        data = Script.substitute_env_vars(data)
        self.assertEqual('string', data)

        data = '--' + os.environ['PFSSPEC_DATA'] + '--'
        data = Script.substitute_env_vars(data)
        self.assertEqual('--${PFSSPEC_DATA}--', data)

        data = ['--' + os.environ['PFSSPEC_ROOT'] + '--', '--' + os.environ['PFSSPEC_DATA'] + '--']
        data = Script.substitute_env_vars(data)
        self.assertEqual(['--${PFSSPEC_ROOT}--', '--${PFSSPEC_DATA}--'], data)

        data = ('--' + os.environ['PFSSPEC_ROOT'] + '--', '--' + os.environ['PFSSPEC_DATA'] + '--')
        data = Script.substitute_env_vars(data)
        self.assertEqual(('--${PFSSPEC_ROOT}--', '--${PFSSPEC_DATA}--'), data)

        data = {'a': '--' + os.environ['PFSSPEC_ROOT'] + '--', 'b': '--' + os.environ['PFSSPEC_DATA'] + '--'}
        data = Script.substitute_env_vars(data)
        self.assertEqual({'a': '--${PFSSPEC_ROOT}--', 'b': '--${PFSSPEC_DATA}--'}, data)

    def test_resolve_env_vars(self):
        data = 1
        data = Script.resolve_env_vars(data)
        self.assertEqual(1, data)

        data = [1, 2]
        data = Script.resolve_env_vars(data)
        self.assertEqual([1, 2], data)

        data = 'string'
        data = Script.resolve_env_vars(data)
        self.assertEqual('string', data)

        data = '--${PFSSPEC_DATA}--'
        data = Script.resolve_env_vars(data)
        self.assertEqual('--' + os.environ['PFSSPEC_DATA'] + '--', data)

        data = ['--${PFSSPEC_ROOT}--', '--${PFSSPEC_DATA}--']
        data = Script.resolve_env_vars(data)
        self.assertEqual(['--' + os.environ['PFSSPEC_ROOT'] + '--', '--' + os.environ['PFSSPEC_DATA'] + '--'], data)

        data = ('--${PFSSPEC_ROOT}--', '--${PFSSPEC_DATA}--')
        data = Script.resolve_env_vars(data)
        self.assertEqual(('--' + os.environ['PFSSPEC_ROOT'] + '--', '--' + os.environ['PFSSPEC_DATA'] + '--'), data)

        data = {'a': '--${PFSSPEC_ROOT}--', 'b': '--${PFSSPEC_DATA}--'}
        data = Script.resolve_env_vars(data)
        self.assertEqual({'a': '--' + os.environ['PFSSPEC_ROOT'] + '--', 'b': '--' + os.environ['PFSSPEC_DATA'] + '--'}, data)