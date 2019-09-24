
import inspect

_figure_data = {}
_figures = {}

def _ensure_nullary(f):
    if f:
        try:
            spec = inspect.getargspec(f)
        except TypeError:
            raise TypeError("{} is not a function".format(f))
        if spec.args:
            raise TypeError("{} should take no arguments".format(f))

def figure_data(setup=None, teardown=None):
    """
    Register a function that creates data for a figure.
    """
    _ensure_nullary(setup)
    _ensure_nullary(teardown)
    def wrapper(wrapped):
        try:
            spec = inspect.getargspec(wrapped)
        except TypeError:
            raise TypeError("{} is not a function".format(wrapped))
        if not spec.args or spec.args[0] != 'exposures':
            raise ValueError("a function registered with figure_data must take at least an `exposures` argument")
        if len(spec.args) - len(spec.defaults) != 1:
            raise ValueError("all secondary arguments must have defaults")
        name = wrapped.__module__[len(__name__)+1:] + '.' + wrapped.__name__
        _figure_data[name] = (wrapped, setup, teardown)
        return wrapped
    return wrapper

def figure(wrapped):
    """
    Register a function that plots a figure
    """
    name = wrapped.__module__[len(__name__)+1:] + '.' + wrapped.__name__
    _figures[name] = wrapped
    return wrapped
