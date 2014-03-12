"""
    H5M Map API low-level operations.

    For Exascale FastForward.
"""

include "config.pxi"

from h5p cimport pdefault, PropID
from h5g cimport GroupID
from h5t cimport typewrap, TypeID

from h5es cimport esid_default, EventStackID
from h5rc cimport RCntxtID
from h5tr cimport TransactionID

from h5py import _objects


def create_ff(GroupID loc not None, char* name, TypeID key_type not None,
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


def open_ff(GroupID loc not None, char* name, RCntxtID rc not None, EventStackID es=None):
    """(GroupID loc, STRING name, RCntxtID rc, EventStackID es=None) => MapID

    Open an existing map object possibly asynchronously.
    """
    cdef hid_t mid
    mid = H5Mopen_ff(loc.id, name, H5P_DEFAULT, rc.id, esid_default(es))
    return MapID.open(mid)


cdef class MapID(ObjectID):
    """ Represents HDF5 map object identifier """

    def _close_ff(self, EventStackID es=None):
        """(EventStackID es=None)

        Close the specified map object, possibly asynchronously.

        Terminate access through this identifier. You shouldn't have to call
        this manually; event stack identifiers are automatically released when
        their Python wrappers are freed.
        """
        with _objects.registry.lock:
            H5Mclose_ff(self.id, esid_default(es))
            if not self.valid:
                del _objects.registry[self.id]


    def get_count_ff(self, RCntxtID rc not None, EventStackID es=None):
        """(RCntxtID rc, EventStackID es=None) => INT count

        Retrieve the number of key/value pairs in the map object, possibly
        asynchronously.
        """
        cdef hsize_t count
        H5Mget_count_ff(self.id, &count, rc.id, esid_default(es))
        return count


    def get_types_ff(self, RCntxtID rc not None, EventStackID es=None):
        """(RCntxtID rc, EventStackID es=None) => TUPLE (TypeID key_type, TypeID val_type)

        Retrieve the datatypes for the keys and values of the map object,
        possibly asynchronously.
        """
        cdef hid_t key_type_id, val_type_id
        H5Mget_types_ff(self.id, &key_type_id, &val_type_id, rc.id, esid_default(es))
        return typewrap(key_type_id), typewrap(val_type_id)
