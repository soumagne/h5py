"""
Python interface to HDF5 Exascale FastForward H5V API
"""

from h5py import h5v, h5s, h5p, h5a, h5f, h5g, h5d, h5q, h5r, h5t, h5m
from h5py import File, Group, Dataset, Datatype, Map
from h5py.query import make_query
from .dataset import readtime_dtype
import numpy


###########################################################################


class Attribute(object):
    """Class representing an HDF5 attribute.

    This is a minimal implementation, just enough to support returning attribute
    objects from views.
    
    For Exascale FastForward.
    """

    @property
    def id(self):
        """ h5a.AttrID instance """
        return self._id


    def __init__(self, bind):
        if not isinstance(bind, h5a.AttrID):
            raise ValueError("%s is not h5a.AttrID" % bind)
        self._id = bind


    def close(self, esid=None):
        """ Close the attribute object """
        self.id._close_ff(es=esid)


    def value(self, rc, esid=None):
        """ Read attribute's value """

        if self.id.get_space().get_simple_extent_type() == h5s.NULL:
            raise IOError("Empty attributes cannot be read")

        tid = self.id.get_type()
        dt = readtime_dtype(self.id.dtype, [])
        val = numpy.ndarray(self.id.shape, dtype=dt, order='C')
        self.id.read_ff(val, rc, es=esid)

        if len(val.shape) == 0:
            return val[()]
        return val


###########################################################################


class View(object):
    """ Represents HDF5 Exascale FastForward H5V API """

    @property
    def id(self):
        """ h5v.ViewID instance """
        return self._id


    def __init__(self, loc, query, rc, sel=None, esid=None):
        """Create a new view object. Arguments:

        loc
            Object on which this view is created. Can be a file, group, or
            dataset.

        query
            Query object, the results of which will be available in this view.

        rc
            Read context object.

        sel
            Optional, default is None. One of selection objects from the
            selections.py module for constraining query results presented in the
            view. Its id property must be of h5s.SpaceID type.

        esid
            Optional, default None. Event stack object.

        For Exascale FastForward.
        """

        if not isinstance(loc.id, (h5g.GroupID, h5d.DatasetID)):
            raise TypeError("A view can only be created on a file, group, or "
                            "dataset")
        if not isinstance(query.id, h5q.QueryID):
            raise TypeError("Query argument must be h5q.QueryID type")

        if sel is None:
            self._id = h5v.create_ff(loc.id, query.id, rc.id, es=esid)
        else:
            if not isinstance(sel.id, h5s.SpaceID):
                raise TypeError("%s is not a h5s.SpaceID" % sel)
            vcpl = h5p.create(h5p.VIEW_CREATE)
            vcpl.set_view_elmt_scope(sel.id)
            self._id = h5v.create_ff(loc.id, query.id, rc.id, vcpl=vcpl,
                                     es=esid)


    def close(self):
        """ Close the view object """
        self.id._close()


    def get_query(self):
        """ Copy of the query object used to create this view """

        qid = self.id.get_query()
        return make_query(qid)


    def count(self):
        """ Count of attributes, objects, and dataset element regions in the
        view. A tuple with three elements is returned.
        """

        return self.id.get_counts()


    def attr_count(self):
        """ Count of attributes in the view. """

        return self.count()[0]


    def obj_count(self):
        """ Count of objects in the view. """
        
        return self.count()[1]


    def dset_count(self):
        """ Count of dataset element regions in the view. """
        
        return self.count()[2]


    def location(self, esid=None):
        """ Object on which the view is created, possibly asynchronously.
        
        For Exascale FastForward.
        """

        locid = self.id.get_location_ff(es=esid)
        if isinstance(locid, h5f.FileID):
            return File(locid)
        elif isinstance(locid, h5g.GroupID):
            return Group(locid)
        elif isinstance(locid, h5d.DatasetID):
            return Dataset(locid)
        else:
            raise ValueError("%s invalid view object location" % locid)


    def attrs(self, start=0, count=1, esid=None):
        """Retrieve the count (default: 1) number of attributes referenced by
        the view object beginning at offset start (default: 0), possibly
        asynchronously.

        This method is a generator of Attribute class objects.

        For Exascale FastForward.
        """

        attrs_id = self.id.get_attrs_ff(start=start, count=count, es=esid)
        for aid in attrs_id:
            yield Attribute(aid)


    def objs(self, start=0, count=1, esid=None):
        """Retrieve the count (default: 1) number of HDF5 objects referenced by
        the view object beginning at offset start (default: 0), possibly
        asynchronously. HDF5 objects returned are: datasets, datatypes, groups,
        and maps.

        This method is a generator of objects of classes Dataset, Datatype,
        Map, and Group.

        For Exascale FastForward.
        """

        objs_id = self.id.get_objs_ff(start=start, count=count, es=esid)
        for oid in objs_id:
            if isinstance(oid, h5d.DatasetID):
                obj = Dataset(oid)
            elif isinstance(oid, h5g.GroupID):
                obj = Group(oid)
            elif isinstance(oid, h5t.TypeID):
                obj = Datatype(oid)
            elif isinstance(oid, h5m.MapID):
                obj = Map(oid)
            else:
                raise ValueError("%s unexpected object type" % oid)

            yield obj


    def regions(self, start=0, count=1, esid=None):
        """Retrieve the count (default: 1) number of region references
        from the view object beginning at offset start (default: 0),
        possibly asynchronously. Each region reference points to a dataset's
        elements included in the view.

        This method is a generator of RegionReference objects.

        For Exascale FastForward.
        """

        tl = self.id.get_elem_regions_ff(start=start, count=count, es=esid)
        for dsid, spaceid in tl:
            yield h5r.create(dsid, '.', h5r.DATASET_REGION, spaceid)
