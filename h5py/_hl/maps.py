# High-level Python interface for Map Exascale FastForward object.

import numpy
import h5py
from h5py import h5s, h5t, h5m
from .base import HLObject



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
            dtype = None
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

    def __init__(self, mapid):
        """ Initialize a new EFF map object for the MapID identifier object
        """
        if not isinstance(mapid, h5m.MapID):
            raise ValueError("%s is not a MapID object" % mapid)
        HLObject.__init__(self, mapid)

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
        self.id._close(esid=esid

    def count(self, rcid, esid=None):
        """Count of key/value pairs.

        For Exascale FastForward.
        """
        return self.id.get_count_ff(rcid, es=esid)

    def __len__(self):
        """Count of key/value pairs."""
        return self.count(self._rcid, esid=self._esid)


