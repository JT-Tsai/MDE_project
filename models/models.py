import copy

models = {}

def register(name):
    """for each model, we define a decorator function
        that takes the model name as an argument and return a class with custom parameters"""
    
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make(model_spec, args = None, load_sd = False):
    """model_spec: initial model specification
        args: a dictionary of custom parameters
        load_sd: whether to load the model state dictionary"""
    
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']

    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model