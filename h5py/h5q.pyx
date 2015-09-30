# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
"""
    Provides access to the low-level HDF5 "H5Q" query interface.
"""

cdef extern from "hdf5.h":
    hid_t H5Qcreate(H5Q_type_t query_type, H5Q_match_op_t match_op, ...) except *

# Compile-time imports
from _objects cimport pdefault
from h5p cimport PropID
from h5t cimport TypeID
from numpy cimport import_array, ndarray, PyArray_DATA
from utils cimport check_numpy_read, emalloc, efree

from h5py import _objects
from ._objects import phil, with_phil

# Initialization
import_array()

# === Public constants and data structures ====================================

REF_REG         = H5Q_REF_REG
REF_OBJ         = H5Q_REF_OBJ
REF_ATTR        = H5Q_REF_ATTR

VIEW_REF_REG_NAME   = H5Q_VIEW_REF_REG_NAME
VIEW_REF_OBJ_NAME   = H5Q_VIEW_REF_OBJ_NAME
VIEW_REF_ATTR_NAME  = H5Q_VIEW_REF_ATTR_NAME

TYPE_DATA_ELEM  = H5Q_TYPE_DATA_ELEM
TYPE_ATTR_VALUE = H5Q_TYPE_ATTR_VALUE
TYPE_ATTR_NAME  = H5Q_TYPE_ATTR_NAME
TYPE_LINK_NAME  = H5Q_TYPE_LINK_NAME
TYPE_MISC       = H5Q_TYPE_MISC

MATCH_EQUAL     = H5Q_MATCH_EQUAL
MATCH_NOT_EQUAL = H5Q_MATCH_NOT_EQUAL
MATCH_LESS_THAN = H5Q_MATCH_LESS_THAN
MATCH_GREATER_THAN = H5Q_MATCH_GREATER_THAN

COMBINE_AND     = H5Q_COMBINE_AND
COMBINE_OR      = H5Q_COMBINE_OR
SINGLETON       = H5Q_SINGLETON

# === Query operations ========================================================

@with_phil
def create(int query_type, int match_op, *args):
    """(INT query_type, INT match_op, *args) => QueryID

    Create a new atomic query object with match_op condition.
    """
    cdef hid_t qid, dtid
    cdef char* name

    if query_type == H5Q_TYPE_DATA_ELEM or query_type == H5Q_TYPE_ATTR_VALUE:
        dt = args[0]
        if not isinstance(dt, TypeID):
            raise ValueError("Third argument must be TypeID")
        dtid = dt.id

        value = args[1]
        if not isinstance(value, ndarray):
            raise ValueError("Fourth argument must be ndarray")
        check_numpy_read(value)

        qid = H5Qcreate(<H5Q_type_t>query_type, <H5Q_match_op_t>match_op, dtid,
                        PyArray_DATA(value))

    elif query_type == H5Q_TYPE_ATTR_NAME or query_type == H5Q_TYPE_LINK_NAME:
        obj_name = args[0]
        if not isinstance(obj_name, str):
            raise ValueError("Third argument must be string")
        name = obj_name

        qid = H5Qcreate(<H5Q_type_t>query_type, <H5Q_match_op_t>match_op, name)

    else:
        raise ValueError("%d: Unsupported query type" % query_type)

    return QueryID(qid)


@with_phil
def decode(buf):
    """(STRING buf) => QueryID

    Unserialize a query.  Bear in mind you can also use the native
    Python pickling machinery to do this.
    """
    cdef char* buf_ = buf
    return QueryID(H5Qdecode(buf_))


cdef class QueryID(ObjectID):

    """ HDF5 query object identifier class """

    @with_phil
    def combine(self, int combine_op, QueryID other not None):
        """(INT combine_op, QueryID other) => QueryID

        Create a new compound query by combining two query objects using the
        operator combine_op.
        """
        cdef hid_t qid
        qid = H5Qcombine(self.id, <H5Q_combine_op_t>combine_op, other.id)
        return QueryID.open(qid)


    @with_phil
    def get_type(self):
        """() => INT type

        Get the query type of the atomic query object.
        """
        cdef H5Q_type_t query_type
        H5Qget_type(self.id, &query_type)
        return <int>query_type


    @with_phil
    def get_match_op(self):
        """() => INT op

        Get the match operator of the atomic query object.
        """
        cdef H5Q_match_op_t match_op
        H5Qget_match_op(self.id, &match_op)
        return <int>match_op


    @with_phil
    def get_components(self):
        """() => TUPLE (QueryID subquery1, QueryID subquery2)

        Get component queries from the compound query.
        """
        cdef hid_t subq1, subq2
        H5Qget_components(self.id, &subq1, &subq2)
        return QueryID.open(subq1), QueryID.open(subq2)


    @with_phil
    def get_combine_op(self):
        """() => INT combine_op

        Get the combine operator type of the compound query object.
        """
        cdef H5Q_combine_op_t combine_op
        H5Qget_combine_op(self.id, &combine_op)
        return <int>combine_op


    @with_phil
    def encode(self):
        """() => STRING

        Serialize a query. Bear in mind you can also use the native Python
        pickling machinery to do this.
        """
        cdef void* buf = NULL
        cdef size_t nalloc = 0

        H5Qencode(self.id, NULL, &nalloc)
        buf = emalloc(nalloc)
        try:
            H5Qencode(self.id, buf, &nalloc)
            pystr = PyBytes_FromStringAndSize(<char*>buf, nalloc)
        finally:
            efree(buf)

        return pystr


    @with_phil
    def apply(self, ObjectID loc not None, PropID vcpl=None):
        """(ObjectID loc, PropID dcpl=None) => TUPLE(group, result)

        Apply query to location and return view (anonymous group).
        """
        import h5i
        cdef unsigned result
        cdef hid_t gid
        gid = H5Qapply(loc.id, self.id, &result, pdefault(vcpl))
        return (h5i.wrap_identifier(gid), result)

