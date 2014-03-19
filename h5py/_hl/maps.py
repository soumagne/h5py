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
