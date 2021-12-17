import argparse
import json
import gzip
import sys
import os
from setuptools import find_packages
from pkgutil import iter_modules
import importlib
import inspect


def find_modules(path):
    """Find all modules below the current path"""
    modules = set()
    for pkg in find_packages(path):
        modules.add(pkg)
        pkgpath = path + "/" + pkg.replace(".", "/")
        if sys.version_info.major == 2 or (
            sys.version_info.major == 3 and sys.version_info.minor < 6
        ):
            for _, name, ispkg in iter_modules([pkgpath]):
                if not ispkg:
                    modules.add(pkg + "." + name)
        else:
            for info in iter_modules([pkgpath]):
                if not info.ispkg:
                    modules.add(pkg + "." + info.name)
    return modules


try:
    import docutils.frontend
    import docutils.utils
    import docutils.parsers.rst
    from docutils import nodes

    class ParamHarvester(nodes.SparseNodeVisitor):
        """Extract Sphinxy parameter descriptions from a docstring."""

        def __init__(self, document):
            nodes.SparseNodeVisitor.__init__(self, document)
            self.tags = set(["param", "parameter", "arg", "argument", "key", "keyword"])

        def visit_document(self, node):
            self.params = {}
            self.field_name = None
            self.field_body = None
            self.field_list = None

        def visit_field_list(self, node):
            self.field_list = node

        def depart_field(self, node):
            if self.field_list is not None and self.field_name is not None:
                self.field_list.remove(node)
            self.field_name = None
            self.field_body = None

        def visit_field_name(self, node):
            fields = node.children[0].split()
            if fields[0] in self.tags and len(fields) > 1:
                self.field_name = fields[1]
            else:
                self.field_name = None
            self.field_body = None

        def visit_field_body(self, node):

            self.field_body = node.astext()
            if len(self.field_body) == 0:
                self.field_body = None

        def depart_field_body(self, node):
            if self.field_name is not None and self.field_body is not None:
                self.params[str(self.field_name)] = self.field_body

    def getdoc(obj):
        docstring = inspect.getdoc(obj)
        if docstring is None:
            return ("", {})

        parser = docutils.parsers.rst.Parser()
        settings = docutils.frontend.OptionParser(
            components=(docutils.parsers.rst.Parser,)
        ).get_default_values()
        doc = docutils.utils.new_document("", settings)
        harvester = ParamHarvester(doc)

        parser.parse(docstring, doc)
        doc.walkabout(harvester)
        return (doc.astext().strip(), harvester.params)


except ImportError:

    def getdoc(obj):
        docstring = inspect.getdoc(obj)
        if docstring is None:
            return ("", {})
        else:
            return (docstring, {})


class DetectorConfigAction(argparse.Action):
    """Add a detector configuration"""

    def __call__(self, parser, args, values, option_string=None):
        if not values or len(values) % 2:
            raise argparse.ArgumentError(
                self, "expected an even number of arguments, got {}".format(len(values))
            )
        config = []
        for detector, livetime in zip(values[::2], values[1::2]):
            try:
                livetime = float(livetime)
            except TypeError:
                raise argparse.ArgumentTypeError(
                    self,
                    "expected pairs of str, float, got {} {}".format(
                        detector, livetime
                    ),
                )
            config.append((detector, livetime))
        getattr(args, self.dest).append(sorted(config))


def jsonify(obj):
    """
    Recursively cast to JSON native types
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    elif hasattr(obj, "keys"):
        return {jsonify(k): jsonify(obj[k]) for k in obj.keys()}
    elif hasattr(obj, "__len__") and not isinstance(obj, str):
        return [jsonify(v) for v in obj]
    else:
        return obj


def make_figure_data():
    from toise import figures

    # find all submodules of toise.figures and import them
    for submod in find_modules(os.path.dirname(sys.modules[__name__].__file__)):
        importlib.import_module(".".join(__name__.split(".")[:-1] + [submod]))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    for name, (func, setup, teardown) in sorted(figures._figure_data.items()):
        docstring, param_help = getdoc(func)
        spec = inspect.getargspec(func)
        p = subparsers.add_parser(
            name, help=docstring, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        p.set_defaults(command=(func, setup, teardown))
        p.add_argument(
            "-d",
            "--detector",
            default=[],
            action=DetectorConfigAction,
            nargs="+",
            help="sequence of detector configuration/livetime pairs",
        )
        p.add_argument("-o", "--outputfile", action="append", help="")
        assert len(spec.args) - (
            (len(spec.defaults) if spec.defaults else 0) == 1
        ), "exposures argument is required"
        _add_options_for_args(p, spec, param_help)
    args = parser.parse_args().__dict__
    exposures = args.pop("detector")
    if not exposures:
        parser.error("at least one detector configuration is required")
    outfiles = args.pop("outputfile")
    if outfiles and len(outfiles) != len(exposures):
        parser.error(
            "either none or all output file names should be specified. got {} output files for {} detector configs".format(
                len(outfiles), len(exposures)
            )
        )

    func, setup, teardown = args.pop("command")
    if setup:
        setup()
    try:
        for exposure, outfile in zip(exposures, outfiles):
            meta = {
                "source": func.__module__ + "." + func.__name__,
                "detectors": exposure,
                "args": args,
            }
            meta["data"] = jsonify(func(tuple(exposure), **args))
            if not outfile.endswith(".json.gz"):
                outfile = outfile + ".json.gz"
            with gzip.open(outfile, "wt") as f:
                json.dump(meta, f, indent=2)
    finally:
        if teardown:
            teardown()


def load_gzip(fname):
    if fname.endswith(".gz"):
        open_file = gzip.open
    else:
        open_file = open
    with open_file(fname) as f:
        return json.load(f)


def _add_options_for_args(parser, spec, param_help):
    num_required_args = 0
    num_required_args = len(spec.args) - (len(spec.defaults) if spec.defaults else 0)
    if spec.defaults:
        for arg, default in zip(spec.args[num_required_args:], spec.defaults):
            argname = arg.replace("_", "-")
            if type(default) is bool:
                parser.add_argument(
                    "--{}".format("no-" if default else "") + argname,
                    default=default,
                    action="store_false" if default else "store_true",
                    dest=arg,
                    help=param_help.get(arg, None),
                )
            else:
                parser.add_argument(
                    "--" + argname,
                    type=type(default),
                    default=default,
                    help=param_help.get(arg, None),
                )


def make_figure():
    from toise import figures

    # find all submodules of toise.figures and import them
    for submod in find_modules(os.path.dirname(sys.modules[__name__].__file__)):
        importlib.import_module(".".join(__name__.split(".")[:-1] + [submod]))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    for name, func in sorted(figures._figures.items()):
        docstring, param_help = getdoc(func)
        spec = inspect.getargspec(func)
        p = subparsers.add_parser(
            name, help=docstring, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        p.set_defaults(command=func)
        p.add_argument("-o", "--outfile")
        spec = inspect.getargspec(func)
        num_required_args = 0
        if (
            spec.args
            and len(spec.args) - (len(spec.defaults) if spec.defaults else 0) == 1
        ):
            p.add_argument("infiles", nargs="+")
            num_required_args = 1
        _add_options_for_args(p, spec, param_help)
    kwargs = parser.parse_args().__dict__
    infiles = kwargs.pop("infiles", None)
    outfile = kwargs.pop("outfile")

    func = kwargs.pop("command")
    if infiles:
        args = (list(map(load_gzip, infiles)),)
    else:
        args = tuple()
    figure = func(*args, **kwargs)

    if outfile:
        figure.savefig(outfile)
    else:
        import matplotlib.pyplot as plt

        plt.show()
