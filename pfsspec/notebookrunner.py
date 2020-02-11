import os
import sys
import logging
from shutil import copyfile
from nbparameterise import extract_parameters, parameter_values, replace_definitions
import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor, ClearMetadataPreprocessor
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from nbconvert import HTMLExporter

class NotebookRunner():
    def __init__(self):
        self.input_notebook = None
        self.working_dir = None
        self.output_notebook = None
        self.output_html = None
        self.parameters = None
        self.kernel = None
        self.nb = None

    def copy_notebook(self):
        copyfile(self.input_notebook, self.output_notebook)

    def open_notebook(self):
        with open(self.output_notebook, 'r') as f:
            self.nb = nbformat.read(f, as_version=4)

    def set_kernel(self):
        if self.kernel is not None:
            ks = self.nb.metadata.get('kernelspec', {})
            ks['name'] = self.kernel

    def set_params(self):
        orig_params = extract_parameters(self.nb)
        new_params = parameter_values(orig_params, **self.parameters)
        self.nb = replace_definitions(self.nb, new_params, execute=False)

    def convert_html(self):
        if self.output_html is not None:
            html_exporter = HTMLExporter()
            #html_exporter.template_file = 'basic'
            (body, resources) = html_exporter.from_notebook_node(self.nb)
            with open(self.output_html, 'w') as f:
                f.write(body)

    def clear_all_output(self):
        cpp = ClearOutputPreprocessor()
        self.nb, resources = cpp.preprocess(self.nb, None)

    def execute_notebook(self):
        resources = {
            'metadata': {
                'path': self.working_dir
            }
        }
        epp = ExecutePreprocessor(timeout=None)
        try:
            self.nb, resources = epp.preprocess(self.nb, resources)
        except CellExecutionError as ex:
            logging.error('An error has occured while execution notebook {}'.format(self.input_notebook))
            # Error is not propagated to allow saving notebook

    def save_notebook(self):
        with open(self.output_notebook, 'w') as f:
            nbformat.write(self.nb, f)

    def run(self):
        self.copy_notebook()
        self.open_notebook()
        self.set_kernel()
        self.set_params()
        self.clear_all_output()
        self.execute_notebook()
        self.save_notebook()
        self.convert_html()
