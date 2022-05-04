from numpy.distutils.core import setup
from numpy.distutils.core import Extension

setup(
    name="nusigma",
    version="1.19",
    packages=["nusigma"],
    package_data={
        "nusigma": ["dat/*.pds", "dat/*.tbl"],
    },
    ext_modules=[
        Extension(
            "nusigma._nusigma",
            ["_nusigma.pyf", "dqagse.f", "d1mach.f"],
            extra_objects=[
                "lib/libnusigma.a",
            ],
        )
    ],
)
