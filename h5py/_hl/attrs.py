# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import numpy
import collections

import h5py
from h5py import h5s, h5t, h5a
from . import base
from .dataset import readtime_dtype


class AttributeManager(base.DictCompat, base.CommonStateObject):

    """
        Allows dictionary-style access to an HDF5 object's attributes.
        For Exascale FastForward.

        These are created exclusively by the library and are available as
        a Python attribute at <object>.attrs

        Like Group objects, attributes provide a minimal dictionary-
        style interface.  Anything which can be reasonably converted to a
        Numpy array or Numpy scalar can be stored.

        Attributes are automatically created on assignment with the
        syntax <obj>.attrs[name] = value, with the HDF5 type automatically
        deduced from the value.  Existing attributes are overwritten.

        To modify an existing attribute while preserving its type, use the
        method modify().  To specify an attribute of a particular type and
        shape, use create().
    """

    def __init__(self, parent):
        """ Private constructor.
        """
        self._id = parent.id

        # For Exascale FastForward. Holds current transaction, read
        # context, and event stack identifier objects.
        self._trid = None
        self._rcid = None
        self._esid = None

    # For Exascale FastForward.
    def set_rc_env(self, rcid, esid=None):
        """Set read context environment to be used. Event stack ID object
        is optional argument, default set to None.

        Note: This is very experimental and may change.
        """
        self._rcid = rcid
        self._esid = esid

    # For Exascale FastForward.
    def set_tr_env(self, trid, esid=None):
        """Set transaction environment to be used. Event stack ID object
        is optional argument, default set to None.

        Note: This is very experimental and may change.
        """
        self._trid = trid
        self._esid = esid

    def __getitem__(self, name):
        """ Read the value of an attribute.

        For Exascale FastForward.
        """
        attr = h5a.open_ff(self._id, self._rcid, self._e(name), es=self._esid)

        if attr.get_space().get_simple_extent_type() == h5s.NULL:
            raise IOError("Empty attributes cannot be read")

        tid = attr.get_type()

        rtdt = readtime_dtype(attr.dtype, [])

        arr = numpy.ndarray(attr.shape, dtype=rtdt, order='C')
        attr.read_ff(arr, self._rcid, es=self._esid)

        if len(arr.shape) == 0:
            return arr[()]
        return arr

    def __setitem__(self, name, value):
        """ Set a new attribute, overwriting any existing attribute.

        The type and shape of the attribute are determined from the data.  To
        use a specific type or shape, or to preserve the type of an attribute,
        use the methods create() and modify().
        """
        self.create(name, data=value, trid=self._trid, dtype=base.guess_dtype(value),
                    esid=self._esid)

    def __delitem__(self, name):
        """ Delete an attribute (which must already exist). """
        h5a.delete_ff(self._id, self._trid, self._e(name), es=self._esid)

    def create(self, name, data, trid, shape=None, dtype=None, esid=None):
        """ Create a new attribute, overwriting any existing attribute.

        For Exascale FastForward.

        name
            Name of the new attribute (required)
        data
            An array to initialize the attribute (required)
        trid
            Transaction identifier object.
        shape
            Shape of the attribute.  Overrides data.shape if both are
            given, in which case the total number of points must be unchanged.
        dtype
            Data type of the attribute.  Overrides data.dtype if both
            are given.
        esid (None)
            Event stack identifier object.
        """

        if data is not None:
            data = numpy.asarray(data, order='C', dtype=dtype)
            if shape is None:
                shape = data.shape
            elif numpy.product(shape) != numpy.product(data.shape):
                raise ValueError("Shape of new attribute conflicts with shape of data")

            if dtype is None:
                dtype = data.dtype

        if isinstance(dtype, h5py.Datatype):
            htype = dtype.id
            dtype = htype.dtype
        else:
            if dtype is None:
                dtype = numpy.dtype('f')
            htype = h5t.py_create(dtype, logical=True)

        if shape is None:
            raise ValueError('At least one of "shape" or "data" must be given')

        data = data.reshape(shape)

        space = h5s.create_simple(shape)

        if name in self:
            h5a.delete_ff(self._id, trid, self._e(name), es=esid)

        attr = h5a.create_ff(self._id, self._e(name), htype, space, trid, es=esid)

        if data is not None:
            try:
                attr.write_ff(data, trid, es=esid)
            except:
                attr._close_ff(es=esid)
                h5a.delete_ff(self._id, trid, self._e(name), es=esid)
                raise

    def modify(self, name, value, trid, esid=None):
        """ Change the value of an attribute while preserving its type.

        For Exascale FastForward. Note that in this method both a read
        context and a transaction are required. For now, the transaction is
        a required argument while the read context should be supplied via
        the set_rc_env() method. This interface may change in the future.

        Differs from __setitem__ in that if the attribute already exists, its
        type is preserved.  This can be very useful for interacting with
        externally generated files.

        If the attribute doesn't exist, it will be automatically created.
        """
        if not name in self:
            self[name] = value
        else:
            value = numpy.asarray(value, order='C')

            attr = h5a.open_ff(self._id, self._rcid, self._e(name), es=esid)

            if attr.get_space().get_simple_extent_type() == h5s.NULL:
                raise IOError("Empty attributes can't be modified")

            # Allow the case of () <-> (1,)
            if (value.shape != attr.shape) and not \
               (numpy.product(value.shape) == 1 and numpy.product(attr.shape) == 1):
                raise TypeError("Shape of data is incompatible with existing attribute")
            attr.write_ff(value, trid, es=esid)

    def __len__(self):
        """ Number of attributes attached to the object. 
        
        For Exascale FastForward.
        """
        # I expect we will not have more than 2**32 attributes
        return h5a.get_num_attrs(self._id)

    def __iter__(self):
        """ Iterate over the names of attributes. """
        attrlist = []

        def iter_cb(name, *args):
            attrlist.append(self._d(name))
        h5a.iterate(self._id, iter_cb)

        for name in attrlist:
            yield name

    def __contains__(self, name):
        """ Determine if an attribute exists, by name.

        For Exascale FastForward.
        """
        return h5a.exists_ff(self._id, self._e(name), self._rcid, es=self._esid)

    def __repr__(self):
        if not self._id:
            return "<Attributes of closed HDF5 object>"
        return "<Attributes of HDF5 object at %s>" % id(self._id)

collections.MutableMapping.register(AttributeManager)
