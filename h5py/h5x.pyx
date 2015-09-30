# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
"""
    Provides access to the low-level HDF5 "H5X" index interface.
"""

# Compile-time imports
from _objects cimport ObjectID, pdefault
from h5p cimport PropID

from h5py import _objects
from ._objects import phil, with_phil

# Initialization

# === Public constants and data structures ====================================

PLUGIN_ERROR    = H5X_PLUGIN_ERROR
PLUGIN_NONE     = H5X_PLUGIN_NONE
PLUGIN_DUMMY    = H5X_PLUGIN_DUMMY
PLUGIN_FASTBIT  = H5X_PLUGIN_FASTBIT
PLUGIN_ALACRITY = H5X_PLUGIN_ALACRITY

PLUGIN_RESERVED = H5X_PLUGIN_RESERVED
PLUGIN_MAX      = H5X_PLUGIN_MAX
MAX_NPLUGINS    = H5X_MAX_NPLUGINS

TYPE_LINK_NAME  = H5X_TYPE_LINK_NAME
TYPE_ATTR_NAME  = H5X_TYPE_ATTR_NAME
TYPE_DATA_ELEM  = H5X_TYPE_DATA_ELEM
TYPE_MAP_VALUE  = H5X_TYPE_MAP_VALUE

# === Index operations ========================================================

@with_phil
def create(ObjectID scope not None, unsigned plugin, PropID xcpl=None):
    """(ObjectID scope, UINT plugin, PropID xcpl=None)

    Create a new index of type plugin in the location, given by scope.
    """
    H5Xcreate(scope.id, plugin, pdefault(xcpl))


@with_phil
def remove(ObjectID scope not None, unsigned plugin):
    """(ObjectID scope, UINT plugin)

    Remove index of type plugin from the location, given by scope.
    """
    H5Xremove(scope.id, plugin)


@with_phil
def get_count(ObjectID scope not None):
    """(ObjectID scope) => INT count

    Number of indexes on a location, given by scope.
    """
    cdef hsize_t idx_count
    H5Xget_count(scope.id, &idx_count)
    return <int>idx_count


@with_phil
def get_size(ObjectID scope not None):
    """(ObjectID scope) => LONG storage_size

    Size allocated for indexes on a location, given by scope.
    """
    return H5Xget_size(scope.id)

