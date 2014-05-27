"""
Python interface to the Exascale FastForward HDF5 H5Q API
"""

from h5py import h5q, h5t
import numpy as np

_query_type = {'data_elem': h5q.TYPE_DATA_ELEM,
               'attr_value': h5q.TYPE_ATTR_VALUE,
               'attr_name': h5q.TYPE_ATTR_NAME,
               'link_name': h5q.TYPE_LINK_NAME}

_match_op = {'=': h5q.MATCH_EQUAL,
             '!=': h5q.MATCH_NOT_EQUAL,
             '<': h5q.MATCH_LESS_THAN,
             '>': h5q.MATCH_GREATER_THAN}


class Query(object):
    """ Mixin class for other query classes """

    @property
    def id(self):
        """ Holds h5q.QueryID object """
        return self._id


    def __repr__(self):
        if not self.id:
            return "<Closed HDF5 query object>"
        else:
            if self.is_atomic():
                qt = 'atomic'
            else:
                qt = 'compound'
        return '<HDF5 %s query object (id: %s)>' % (qt, str(self.id.id))


    def is_atomic(self):
        cop = self.id.get_combine_op()
        return cop == h5q.SINGLETON


    def is_compound(self):
        cop = self.id.get_combine_op()
        return cop != h5q.SINGLETON


def _val_helper(obj):
    """ Returns a tuple with ndarray and TypeID representation of obj. """

    val = np.asarray(obj, order='C')
    dt = h5t.py_create(val.dtype, logical=1)

    return val, dt


class AtomicQuery(Query):
    """ Atomic query class """

    def __init__(self, qobj):
        self._id = None

        if isinstance(qobj, h5q.QueryID):
            self._id = qobj
        else:
            if qobj not in _query_type:
                raise ValueError("%s: Invalid query type" % qobj)
            self._qt = qobj


    def __gt__(self, other):
        if self.id is not None:
            raise RuntimeError("Atomic query already created")
        if self._qt not in ('data_elem', 'attr_value'):
            raise NotImplementedError("Invalid operation for %s query type"
                                      % self._qt)
        qt = _query_type[self._qt]
        val, dt = _val_helper(other)
        qid = h5q.create(qt, h5q.MATCH_GREATER_THAN, dt, val)
        self._id = qid
        
        return self


    def __lt__(self, other):
        if self.id is not None:
            raise RuntimeError("Atomic query already created")
        if self._qt not in ('data_elem', 'attr_value'):
            raise NotImplementedError("Invalid operation for %s query type"
                                      % self._qt)
        qt = _query_type[self._qt]
        val, dt = _val_helper(other)
        qid = h5q.create(qt, h5q.MATCH_LESS_THAN, dt, val)
        self._id = qid
        
        return self


    def __eq__(self, other):
        if self.id is not None:
            raise RuntimeError("Atomic query already created")
        if self._qt in ('data_elem', 'attr_value'):
            qt = _query_type[self._qt]
            val, dt = _val_helper(other)
            qid = h5q.create(qt, h5q.MATCH_EQUAL, dt, val)
            self._id = qid
        elif self._qt in ('attr_name', 'link_name'):
            name = str(other)
            qt = _query_type[self._qt]
            qid = h5q.create(qt, h5q.MATCH_EQUAL, name)
            self._id = qid

        return self


    def __neq__(self, other):
        if self.id is not None:
            raise RuntimeError("Atomic query already created")
        if self._qt in ('data_elem', 'attr_value'):
            qt = _query_type[self._qt]
            val, dt = _val_helper(other)
            qid = h5q.create(qt, h5q.MATCH_NOT_EQUAL, dt, val)
            self._id = qid
        elif self._qt in ('attr_name', 'link_name'):
            name = str(other)
            qt = _query_type[self._qt]
            qid = h5q.create(qt, h5q.MATCH_NOT_EQUAL, name)
            self._id = qid

        return self


    def __and__(self, other_query):
        qid = h5q.combine(self.id, h5q.COMBINE_AND, other_query.id)
        return CompoundQuery(qid)


    def __or__(self, other_query):
        qid = h5q.combine(self.id, h5q.COMBINE_OR, other_query.id)
        return CompoundQuery(qid)


    def query_type(self):
        """ Returns string representation of the query's type. """
        
        qt = self.id.get_query_type()
        return _query_type.keys()[_query_type.values().index(qt)]


    def match_op(self):
        """ Returns string representation of the query's match operator. """
        
        op = self.id.get_match_op()
        return _match_op.keys()[_match_op.values().index(op)]


class CompoundQuery(Query):
    """ Compound query class """

    def __init__(self, qid):
        if isinstance(qid, h5q.QueryID):
            raise ValueError("%s is not QueryID" % qid)
        self._id = qid


    def components(self):
        """ Tuple of the subqueries of the compound query. """
        
        qtpl = self.id.get_components()
        qlist = []
        for qid in qtpl:
            if qid.get_combine_op() == h5q.SINGLETON:
                # Subquery is atomic...
                qlist.append(AtomicQuery(qid))
            else:
                # Subquery is compound...
                qlist.append(CompoundQuery(qid))

        return tuple(qlist)


    def combine_op(self):
        """ Returns string representation of the compound query's combine
        operator. """

        op = self.id.get_combine_op()
        if op == h5q.COMBINE_AND:
            return '&'
        elif op == h5q.COMBINE_OR:
            return '|'
        else:
            raise RuntimeError("%s: Invalid combine operator for compound query"
                               % op)
