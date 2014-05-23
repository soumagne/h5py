"""
H5V API Low-level Bindings

For Exascale FastForward.
"""

include "config.pxi"

from _objects cimport ObjectID
from utils cimport emalloc, efree
from h5p cimport pdefault, PropCreateID
from h5py cimport h5i
from h5tr cimport TransactionID
from h5rc cimport RCntxtID
from h5es cimport esid_default, EventStackID
from h5q cimport QueryID

from h5py import h5
if not h5.get_config().eff:
    raise RuntimeError("HDF5 library not built with FastForward support")


# H5V API Bindings

def create_ff(ObjectID loc not None, QueryID q not None, RCntxtID rc not None,
              PropCreateID vcpl=None, EventStackID es=None):
    """(ObjectID loc, QueryID q, RCntxtID rc, PropCreateID vcpl=None, EventStackID es=None) => ViewID view

    Create a new view object on the object loc, possibly asynchronously. loc can
    be either a container, group, or dataset.

    For Exascale FastForward.
    """
    cdef hid_t vid
    vid = H5Vcreate_ff(loc.id, q.id, pdefault(vcpl), rc.id, esid_default(es))
    return ViewID.open(vid)


cdef class ViewID(ObjectID):
    """ Represents HDF5 Exacale FastForward view object """

    def get_query(self):
        """() => QueryID

        Get a copy of the query used to create this view.
        """
        cdef hid_t qid
        H5Vget_query(self.id, &qid)
        return QueryID.open(qid)


    def get_counts(self):
        """() => TUPLE (INT attr_count, INT obj_count, INT region_count)

        Get the number of attributes, objects, and dataset element regions in
        the view.
        """
        cdef hsize_t attr_count, obj_count, region_count
        H5Vget_counts(self.id, &attr_count, &obj_count, &region_count)
        return (<int>attr_count, <int>obj_count, <int>region_count)


    def get_location_ff(self, EventStackID es=None):
        """(EventStackID es=None) => ObjectID

        Get the location object for this view, possibly asynchronously.

        For Exacale FastForward.
        """
        cdef hid_t locid
        H5Vget_location_ff(self.id, &locid, esid_default(es))
        return h5i.wrap_identifier(locid)


    def get_attrs_ff(self, start=0, count=1, EventStackID es=None):
        """(INT start=0, INT count=1, EventStackID es=None) => LIST

        Retrieve the count (default: 1) number of attributes referenced by the
        view object beginning at offset start (default: 0), possibly
        asynchronously.

        For Exacale FastForward.
        """
        cdef hid_t *attr_id = NULL
        cdef list attr_objs = []
        cdef int i

        try:
            attr_ids = <hid_t*>emalloc(sizeof(hid_t)*count)
            H5Vget_attrs_ff(self.id, <hsize_t>start, <hsize_t>count, attr_id,
                            esid_default(es))
            for i from 0 <= i < count:
                attr_objs.append(h5i.wrap_identifier(attr_id[i]))

            return attr_objs
        finally:
            efree(attr_id)
