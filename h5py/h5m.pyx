"""
    H5M Map API low-level operations.

    For Exascale FastForward.
"""

include "config.pxi"

from h5p cimport pdefault, PropID
from h5g cimport GroupID
from h5t cimport TypeID

from h5es cimport esid_default, EventStackID
from h5rc cimport RCntxtID
from h5tr cimport TransactionID


def create(GroupID loc not None, char* name, TypeID key_type not None,
           TypeID val_type not None, TransactionID tr not None,
           PropID lcpl=None, EventStackID es=None):
    """(GroupID loc, STRING name, TypeID key_type, TypeID val_type,
    TransactionID tr, PropID lcpl=None, EventStackID es=None) => MapID

    Create a new map object and link it into a file or a group, possibly
    asynchronously.
    """
    cdef hid_t mid
    mid = H5Mcreate_ff(loc.id, name, key_type.id, val_type.id, pdefault(lcpl),
                       H5P_DEFAULT, H5P_DEFAULT, tr.id, esid_default(es))
    return MapID.open(mid)
