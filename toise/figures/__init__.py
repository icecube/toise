import inspect
from typing import Sequence

_figure_data = {}
_figures = {}
_tables = {}


def _ensure_nullary(f):
    if isinstance(f, Sequence):
        for element in f:
            _ensure_nullary(element)
        return
    if f:
        try:
            spec = inspect.signature(f)
        except TypeError:
            raise TypeError("{} is not a function".format(f))
        if spec.parameters:
            raise TypeError("{} should take no arguments".format(f))


def figure_data(setup=None, teardown=None):
    """
    Register a function that creates data for a figure.
    """
    _ensure_nullary(setup)
    _ensure_nullary(teardown)

    def wrapper(wrapped):
        try:
            spec = inspect.signature(wrapped)
        except TypeError:
            raise TypeError("{} is not a function".format(wrapped))
        if not spec.parameters or list(spec.parameters.keys())[0] != "exposures":
            raise ValueError(
                "a function registered with figure_data must take at least an `exposures` argument"
            )
        if (
            sum(1 for param in spec.parameters.values() if param.default is param.empty)
            > 1
        ):
            raise ValueError("all secondary arguments must have defaults")
        name = wrapped.__module__[len(__name__) + 1 :] + "." + wrapped.__name__
        _figure_data[name] = (wrapped, setup, teardown)
        return wrapped

    return wrapper


def figure(wrapped):
    """
    Register a function that plots a figure
    """
    name = wrapped.__module__[len(__name__) + 1 :] + "." + wrapped.__name__
    _figures[name] = wrapped
    return wrapped


def table(wrapped):
    """
    Register a function that makes a table
    """
    name = wrapped.__module__[len(__name__) + 1 :] + "." + wrapped.__name__
    _tables[name] = wrapped
    return wrapped
