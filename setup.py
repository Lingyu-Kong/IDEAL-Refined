from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "ideal.models.m3gnet.utils.threebody_indices",
        ["src/ideal/models/m3gnet/utils/threebody_indices.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(ext_modules=cythonize(extensions))
