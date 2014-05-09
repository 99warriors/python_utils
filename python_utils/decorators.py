import functools

class fxn_decorator(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, f):
        
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            raise NotImplementedError

        return wrapped_f

class decorated_method(object):
    
    def __init__(self, f, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, inst, *args, **kwargs):
        """
        this is the new class method.  inst is the instance on which the method is called
        """
        raise NotImplementedError

    def __get__(self, inst, cls):
        return functools.partial(self.__call__, inst)

class method_decorator(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, f):
        return decorated_method(f)
