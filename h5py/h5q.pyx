# H5Q API Low-Level Bindings
include "config.pxi"

cdef extern from "hdf5.h":
    hid_t H5Qcreate(H5Q_type_t query_type, H5Q_match_op_t match_op, ...) except *

include "_locks.pxi"

from _errors cimport set_exception
from h5t cimport typewrap, py_create, TypeID
from numpy cimport import_array, ndarray, PyArray_DATA
from utils cimport check_numpy_read, check_numpy_write

from h5py import _objects

# API Constants

TYPE_DATA_ELEM = H5Q_TYPE_DATA_ELEM
TYPE_ATTR_VALUE = H5Q_TYPE_ATTR_VALUE
TYPE_ATTR_NAME = H5Q_TYPE_ATTR_NAME
TYPE_LINK_NAME = H5Q_TYPE_LINK_NAME

MATCH_EQUAL = H5Q_MATCH_EQUAL
MATCH_NOT_EQUAL = H5Q_MATCH_NOT_EQUAL
MATCH_LESS_THAN = H5Q_MATCH_LESS_THAN
MATCH_GREATER_THAN = H5Q_MATCH_GREATER_THAN

COMBINE_AND = H5Q_COMBINE_AND
COMBINE_OR = H5Q_COMBINE_OR
SINGLETON = H5Q_SINGLETON

# API Bindings

def create(int query_type, int match_op, *args):
    """(INT query_type, INT match_op, *args) => QueryID

    Create a new query_type object with match_op condition.
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

    return QueryID.open(qid)


cdef class QueryID(ObjectID):
    """ HDF5 query object identifier class """

    def _close(self):
        """()

        Close the query object.
        """
        with _objects.registry.lock:
            H5Qclose(self.id)
            if not self.valid:
                del _objects.registry[self.id]


    def combine(self, int combine_op, QueryID other not None):
        """(INT combine_op, QueryID other) => QueryID

        Create a new compound query by combining two query objects using the
        operator combine_op.
        """
        cdef hid_t qid
        qid = H5Qcombine(self.id, <H5Q_combine_op_t>combine_op, other.id)
        return QueryID.open(qid)


    def get_match_type(self):
        """() => INT type

        Get the match type of the atomic query object.
        """
        cdef H5Q_type_t match_type
        H5Qget_match_info(self.id, &match_type, NULL)
        return <int>match_type


    def get_match_op(self):
        """() => INT op

        Get the match operator of the atomic query object.
        """
        cdef H5Q_match_op_t match_op
        H5Qget_match_info(self.id, NULL, &match_op)
        return <int>match_op


    def get_components(self):
        """() => TUPLE subquery1, subquery2

        Get component queries from the compound query.
        """
        cdef hid_t subq1, subq2
        H5Qget_components(self.id, &subq1, &subq2)
        return QueryID.open(subq1), QueryID.open(subq2)


    def get_combine_op(self):
        """() => INT combine_op

        Get the combine operator type of the query object.
        """
        cdef H5Q_combine_op_t combine_op
        H5Qget_combine_op(self.id, &combine_op)
        return <int>combine_op
