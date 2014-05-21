# H5Q API Low-Level Bindings

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

def create_value_query(int query_type, int match_op, TypeID dt not None,
                       ndarray value not None):
    """(INT query_type, INT match_op, TypeID dt, NDARRAY value) => QueryID

    Create a new query_type object with match_op condition for selecting data
    elements or attribute values.
    """
    cdef hid_t qid
    check_numpy_read(value)
    qid = H5Qcreate_v(<H5Q_type_t>query_type, <H5Q_match_op_t>match_op, dt.id,
                      PyArray_DATA(value))
    return QueryID.open(qid)


def create_name_query(int query_type, int match_op, char* name):
    """(INT query_type, INT match_op, STRING name) => QueryID

    Create a new query_type object with match_op condition for selecting
    objects by their name.
    """
    cdef hid_t qid
    qid = H5Qcreate_n(<H5Q_type_t>query_type, <H5Q_match_op_t>match_op, name)
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
