# High-level Python interface for Map Exascale FastForward object.

import numpy
import h5py
from h5py import h5s, h5t, h5m
from .base import HLObject
from .dataset import readtime_dtype
from . import datatype


def make_new_map(parent, name, trid, kdt=None, vdt=None, esid=None):
    """ Create a new map object and return its low-level identifier object

    For Exascale FastForward.
    """

    def type_sel(dt):
        """ Select the data type for key or value """
        if isinstance(dt, h5py.Datatype):
            # This is named datatype, use it...
            return dt.id
        else:
            # More work required to figure out datatype...
            if dt is None:
                dtype = numpy.dtype("=f4")
            else:
                dtype = numpy.dtype(dt)
            return h5t.py_create(dtype, logical=1)

    k_tid = type_sel(kdt)
    v_tid = type_sel(vdt)
    mapid = h5m.create_ff(parent.id, name, k_tid, v_tid, trid, es=esid)
    return mapid


class Map(HLObject):
    """ Represents an Exascale FastForward Map object """

    @property
    def key_dtype(self):
        """Key datatype

        For Exascale FastForward.
        """
        return datatype.Datatype(self.id.key_typeid)

    @property
    def val_dtype(self):
        """Value datatype

        For Exascale FastForward.
        """
        return datatype.Datatype(self.id.val_typeid)

    @property
    def key_shape(self):
        """ Numpy ndarray shape tuple for map's keys """
        return self._key_shape

    @property
    def val_shape(self):
        """ Numpy ndarray shape tuple for map's values """
        return self._val_shape

    def __init__(self, mapid):
        """ Initialize a new EFF map object for the MapID identifier object
        """
        if not isinstance(mapid, h5m.MapID):
            raise ValueError("%s is not a MapID object" % mapid)
        HLObject.__init__(self, mapid)

        # Shapes for storing key and value data...
        self._key_shape = None
        self._val_shape = None

        # For Exascale FastForward. Holds current transaction, read
        # context, and event stack identifier objects.
        self._trid = None
        self._rcid = None
        self._esid = None

    def set_rc_env(self, rcid, esid=None):
        """Set read context environment to be used. Event stack ID object
        is optional argument, default set to None.

        For Exascale FastForward.

        Note: This is very experimental and may change.
        """
        self._rcid = rcid
        self._esid = esid

    def set_tr_env(self, trid, esid=None):
        """Set transaction environment to be used. Event stack ID object
        is optional argument, default set to None.

        For Exascale FastForward.

        Note: This is very experimental and may change.
        """
        self._trid = trid
        self._esid = esid

    def close(self, esid=None):
        """Close the map. Named argument esid (default: None) holds the
        EventStackID identifier.

        For Exascale FastForward.
        """
        self.id._close_ff(es=esid)

    def count(self, rcid, esid=None):
        """Count of key/value pairs.

        For Exascale FastForward.
        """
        return self.id.get_count_ff(rcid, es=esid)

    def __len__(self):
        """Count of key/value pairs.

        For Exascale FastForward.
        """
        #return self.count(self._rcid, esid=self._esid)
        raise NotImplementedError("__len__() not supported, use Map.count()")

    def get(self, key, rcid, esid=None):
        """Read the value for the given key.

        For Exascale FastForward.
        """
        if not self.exists(key, rcid, esid=esid):
            raise KeyError(str(key))
        vdt = readtime_dtype(self.val_dtype.dtype, [])
        val = numpy.ndarray(self.val_shape, dtype=vdt, order='C')
        key = numpy.asarray(key, order='C', dtype=self.key_dtype.dtype)
        self.id.get_ff(key, val, rcid, es=esid)
        if len(val.shape) == 0:
            return val[()]
        return val

    def __getitem__(self, key):
        """Get the value for the given map's key.

        For Exascale FastForward.
        """
        #return self.get(key, self._rcid, esid=self._esid)
        raise NotImplementedError("__getitem__() not supported, use Map.get()")

    def set(self, key, value, trid, esid=None):
        """Set the value for the given key of the map object.

        For Exascale FastForward.
        """
        key = numpy.asarray(key, order='C', dtype=self.key_dtype.dtype)
        value = numpy.asarray(value, order='C', dtype=self.val_dtype.dtype)

        if self.key_shape is None:
            # Remember the key's shape the first time
            self._key_shape = key.shape
        elif self._key_shape != key.shape:
            raise ValueError("Key shape mismatch; got %s, expected %s" %
                             (key.shape, self._key_shape))

        if self.val_shape is None:
            # Remember the value's shape the first time
            self._val_shape = value.shape
        elif self._val_shape != value.shape:
            raise ValueError("Value shape mismatch; got %s, expected %s" %
                             (value.shape, self._val_shape))

        self.id.set_ff(key, value, trid, es=esid)

    def __setitem__(self, key, value):
        """Set the value of the map's key"""
        # self.set(key, value, self._trid, esid=self._esid)
        raise NotImplementedError("__setitem__ not supported, use Map.set()")

    def key_type(self, rcid, esid=None):
        """Return map's key datatype

        For Exascale FastForward.
        """
        t = self.id.get_types_ff(rcid, es=esid)
        return datatype.Datatype(t[0])

    def value_type(self, rcid, esid=None):
        """Return map's value datatype

        For Exascale FastForward.
        """
        t = self.id.get_types_ff(rcid, es=esid)
        return datatype.Datatype(t[1])

    def delete(self, key, trid, esid=None):
        """Delete a key/value pair from the map object.

        For Exascale FastForward.
        """
        key = numpy.asarray(key, order='C', dtype=self.key_dtype.dtype)
        self.id.delete_ff(key, trid, es=esid)

    def __delitem__(self, key):
        """ Delete the map's key """
        # self.delete(key, self._trid, esid=self._esid)
        raise \
            NotImplementedError("__delitem__() not supported, use Map.delete()")

    def exists(self, key, rcid, esid=None):
        """Determine whether a key exists in the map object.

        For Exascale FastForward.
        """
        key = numpy.asarray(key, order='C', dtype=self.key_dtype.dtype)
        return self.id.exists_ff(key, rcid, es=esid)

    def __contains__(self, key):
        """Test if key is in the map"""
        # return self.exists(key, self._rcid, esid=self._esid)
        raise \
            NotImplementedError("__contains__() not supported, use Map.exists()")
