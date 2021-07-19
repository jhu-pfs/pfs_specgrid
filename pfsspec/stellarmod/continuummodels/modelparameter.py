class ModelParameter():
    def __init__(self, name=None, 
        rbf_method=None, rbf_function=None, rbf_epsilon=None,
        orig=None):

        if isinstance(orig, ModelParameter):
            self.name = name or orig.name
            self.rbf_method = rbf_method or orig.rbf_method
            self.rbf_function = rbf_function or orig.rbf_function
            self.rbf_epsilon = rbf_epsilon or orig.rbf_epsilon
        else:
            self.name = name
            self.rbf_method = rbf_method
            self.rbf_function = rbf_function
            self.rbf_epsilon = rbf_epsilon