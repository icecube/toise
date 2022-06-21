import os
import numpy as np
from subprocess import check_call, PIPE
from os import path, unlink, environ, mkdir
from lazy_object_proxy import Proxy
from functools import partial


def defer(func, *args, **kwargs):
    """
    Defer function invocation until an attribute is accessed
    """
    return Proxy(partial(func, *args, **kwargs))


data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "data"))


def refresh_data_tables(icecube_password=os.environ.get("ICECUBE_PASSWORD", None)):
    if icecube_password is None:
        raise EnvironmentError(
            "You need to set the environment variable ICECUBE_PASSWORD to the icecube user password."
        )

    cwd = data_dir
    check_call(
        [
            "curl",
            "--fail",
            "-u",
            "icecube:" + icecube_password,
            "-O",
            "http://convey.icecube.wisc.edu/data/user/jvansanten/projects/2015/gen2_benchmark/data/archive.tar.gz",
        ],
        cwd=cwd,
    )
    check_call(["tar", "xzf", "archive.tar.gz"], cwd=cwd)
    unlink(path.join(data_dir, "archive.tar.gz"))


def center(x):
    return 0.5 * (np.array(x)[1:] + np.array(x)[:-1])


def edge(x):
    """returns bin edges with centers that match x. "inverse" of center(x)."""
    c = center(x)
    return np.concatenate(([2 * x[0] - c[0]], c, [2 * x[-1] - c[-1]]))


class constants:
    second = 1.
    day = 24 * 3600.0
    annum = 365 * day
    cm = 1e2
    cm2 = 1e4
    Mpc = 1e6 #use pc as default units


class baseEnum(int):
    name = None
    values = {}

    def __repr__(self):
        return self.name


class metaEnum(type):
    """Helper metaclass to return the class variables as a dictionary " """

    def __new__(cls, classname, bases, classdict):
        """Return a new class with a "values" attribute filled"""

        newdict = {"values": {}}

        for k in list(classdict.keys()):
            if not (k.startswith("_") or k == "name" or k == "values"):
                val = classdict[k]
                member = baseEnum(val)
                member.name = k
                newdict["values"][val] = member
                newdict[k] = member

        # Tell each member about the values in the enum
        for k in list(newdict["values"].keys()):
            newdict["values"][k].values = newdict["values"]
        # Return a new class with the "values" attribute filled
        return type.__new__(cls, classname, bases, newdict)


class enum(baseEnum, metaclass=metaEnum):
    """This class mimicks the interface of boost-python-wrapped enums.

    Inherit from this class to construct enumerated types that can
    be passed to the I3Datatype, e.g.:

        class DummyEnummy(tableio.enum):
                Foo = 0
                Bar = 1
                Baz = 2

        desc = tableio.I3TableRowDescription()
        desc.add_field('dummy', tableio.I3Datatype(DummyEnummy), '', '')"""


class PDGCode(enum):
    PPlus = 2212
    He4Nucleus = 1000020040
    N14Nucleus = 1000070140
    O16Nucleus = 1000080160
    Al27Nucleus = 1000130270
    Fe56Nucleus = 1000260560
    NuE = 12
    NuEBar = -12
    NuMu = 14
    NuMuBar = -14
    NuTau = 16
    NuTauBar = -16
