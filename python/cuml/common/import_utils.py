#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import importlib
from cuml.common.device_support import BUILT_WITH_CUDA
from distutils.version import LooseVersion


class UnavailableImportError(Exception):
    '''Error thrown if a symbol is unavailable due to an issue importing it'''


class MissingImport:
    '''A placeholder class for unimportable symbols

    This class allows errors to be deferred from import time to the time
    that a symbol is actually used in order to streamline the usage of
    optional dependencies. This is particularly useful for attempted imports
    of GPU-only modules which will only be invoked if GPU-only
    functionality is specifically used.

    Parameters
    ----------
    symbol: str
        The name of the symbol to be used in constructing error messages.
    msg: str or None
        An error message to be displayed if this symbol is directly used.
    '''
    def __init__(self, symbol, msg=None):
        if msg is None:
            self._msg = f'{symbol} could not be imported'
        else:
            self._msg = msg

    def __getattr__(self, name):
        raise UnavailableImportError(self._msg)

    def __call__(self, *args, **kwargs):
        raise UnavailableImportError(self._msg)


def safe_import(module, msg=None):
    '''A function used to import modules that may not be available

    This function will attempt to import a module with the given name, but it
    will not throw an ImportError if the module is not found. Instead, it will
    return a placeholder object which will raise an exception only if used.

    Parameters
    ----------
    module: str
        The name of the module to import.
    msg: str or None
        An optional error message to be displayed if this module is used
        after a failed import.

    Returns
    -------
    object
        The imported module or a MissingImport object.
    '''
    try:
        return importlib.import_module(module)
    except Exception:
        return MissingImport(module, msg=msg)


def safe_import_from(module, symbol, msg=None):
    '''A function used to import symbols from modules that may not be available

    This function will attempt to import a symbol with the given name from
    the given module, but it will not throw an ImportError if the symbol is not
    found. Instead, it will return a placeholder object which will raise an
    exception only if used.

    Parameters
    ----------
    module: str
        The name of the module in which the symbol is defined.
    symbol: str
        The name of the symbol to import.
    msg: str or None
        An optional error message to be displayed if this symbol is used
        after a failed import.

    Returns
    -------
    object
        The imported module or a MissingImport object.
    '''
    try:
        imported_module = importlib.import_module(module)
        return getattr(imported_module, symbol)
    except Exception:
        return MissingImport(".".join((module, symbol)), msg=msg)


def gpu_only_import(self, module):
    '''A function used to import modules required only in GPU installs

    This function will attempt to import a module with the given name, but it
    will only throw an ImportError if the attempt fails AND this is not a
    CPU-only build. This allows GPU-only dependencies to be cleanly
    imported in CPU-only builds but guarantees that the correct exception
    will be raised if a required dependency is unavailable. If the import
    fails on a CPU-only build, a placeholder object will be returned which
    raises an exception only if used.

    Parameters
    ----------
    module: str
        The name of the module to import.

    Returns
    -------
    object
        The imported module or a MissingImport object.
    '''
    if BUILT_WITH_CUDA:
        return importlib.import_module(module)
    else:
        return safe_import(
            module,
            f'{module} is not installed in CPU-only installations'
        )


def gpu_only_import_from(self, module, symbol):
    '''A function used to import symbols required only in GPU installs

    This function will attempt to import a symbol from a module with the given
    names, but it will only throw an ImportError if the attempt fails AND this
    is not a CPU-only build. This allows GPU-only dependencies to be cleanly
    imported in CPU-only builds but guarantees that the correct exception will
    be raised if a required dependency is unavailable. If the import fails on a
    CPU-only build, a placeholder object will be returned which raises an
    exception only if used.

    Parameters
    ----------
    module: str
        The name of the module to import.
    symbol: str
        The name of the symbol to import.

    Returns
    -------
    object
        The imported symbol or a MissingImport object.
    '''
    if BUILT_WITH_CUDA:
        imported_module = importlib.import_module(module)
        return getattr(imported_module, symbol)
    else:
        return safe_import_from(
            module,
            symbol,
            f'{module}.{symbol} is not available in CPU-only installations'
        )


def cpu_only_import(self, module):
    '''A function used to import modules required only in CPU installs

    This function will attempt to import a module with the given name, but it
    will only throw an ImportError if the attempt fails AND this is not a
    GPU-only build. This allows CPU-only dependencies to be cleanly
    imported in GPU-only builds but guarantees that the correct exception
    will be raised if a required dependency is unavailable. If the import
    fails on a GPU-only build, a placeholder object will be returned which
    raises an exception only if used.

    Parameters
    ----------
    module: str
        The name of the module to import.

    Returns
    -------
    object
        The imported module or a MissingImport object.
    '''
    if BUILT_WITH_CUDA:
        return safe_import(
            module,
            f'{module} is not installed in GPU-only installations'
        )
    else:
        return importlib.import_module(module)


def cpu_only_import_from(self, module, symbol):
    '''A function used to import symbols required only in CPU installs

    This function will attempt to import a symbol from a module with the given
    names, but it will only throw an ImportError if the attempt fails AND this
    is not a GPU-only build. This allows CPU-only dependencies to be cleanly
    imported in GPU-only builds but guarantees that the correct exception will
    be raised if a required dependency is unavailable. If the import fails on a
    GPU-only build, a placeholder object will be returned which raises an
    exception only if used.

    Parameters
    ----------
    module: str
        The name of the module to import.
    symbol: str
        The name of the symbol to import.

    Returns
    -------
    object
        The imported symbol or a MissingImport object.
    '''
    if BUILT_WITH_CUDA:
        return safe_import_from(
            module,
            symbol,
            f'{module}.{symbol} is not available in GPU-only installations'
        )
    else:
        imported_module = importlib.import_module(module)
        return getattr(imported_module, symbol)


numba = gpu_only_import('numba')


def has_dask():
    try:
        import dask   # NOQA
        import dask.distributed   # NOQA
        import dask.dataframe   # NOQA
        return True
    except ImportError:
        return False


def has_dask_cudf():
    try:
        import dask_cudf # NOQA
        return True
    except ImportError:
        return False


def has_dask_sql():
    try:
        import dask_sql # NOQA
        return True
    except ImportError:
        return False


def has_cupy():
    try:
        import cupy   # NOQA
        return True
    except ImportError:
        return False


def has_ucp():
    try:
        import ucp  # NOQA
        return True
    except ImportError:
        return False


def has_umap():
    try:
        import umap  # NOQA
        return True
    except ImportError:
        return False


def has_lightgbm():
    try:
        import lightgbm  # NOQA
        return True
    except ImportError:
        return False


def has_xgboost():
    try:
        import xgboost  # NOQA
        return True
    except ImportError:
        return False
    except Exception as ex:
        import warnings
        warnings.warn(
            ("The XGBoost library was found but raised an exception during "
             "import. Importing xgboost will be skipped. "
             "Error message:\n{}").format(str(ex)))
        return False


def has_pytest_benchmark():
    try:
        import pytest_benchmark  # NOQA
        return True
    except ImportError:
        return False


def check_min_dask_version(version):
    try:
        import dask
        return LooseVersion(dask.__version__) >= LooseVersion(version)
    except ImportError:
        return False


def check_min_numba_version(version):
    return LooseVersion(str(numba.__version__)) >= LooseVersion(version)


def check_min_cupy_version(version):
    if has_cupy():
        import cupy
        return LooseVersion(str(cupy.__version__)) >= LooseVersion(version)
    else:
        return False


def has_scipy(raise_if_unavailable=False):
    try:
        import scipy   # NOQA
        return True
    except ImportError:
        if not raise_if_unavailable:
            return False
        else:
            raise ImportError("Scipy is not available.")


def has_sklearn():
    try:
        import sklearn   # NOQA
        return True
    except ImportError:
        return False


def has_hdbscan_plots(raise_if_unavailable=True):
    try:
        from hdbscan.plots import SingleLinkageTree  # NOQA
        return True
    except ImportError:
        if(raise_if_unavailable):
            raise ImportError("hdbscan must be installed to use plots.")
        else:
            return False


def has_shap(min_version="0.37"):
    try:
        import shap  # noqa
        if min_version is None:
            return True
        else:
            return (LooseVersion(str(shap.__version__)) >=
                    LooseVersion(min_version))
    except ImportError:
        return False


def has_daskglm(min_version=None):
    try:
        import dask_glm  # noqa
        if min_version is None:
            return True
        else:
            return (LooseVersion(str(dask_glm.__version__)) >=
                    LooseVersion(min_version))
    except ImportError:
        return False


def dummy_function_always_false(*args, **kwargs):
    return False


class DummyClass(object):
    pass
