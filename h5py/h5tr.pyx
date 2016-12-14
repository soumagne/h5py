# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
"""
    Provides access to the low-level HDF5 "H5TR" transaction interface.
"""

# Compile-time imports
from _objects cimport pdefault

from h5py import _objects
from ._objects import phil, with_phil

# === Public constants and data structures ====================================

# === Transaction operations ========================================================

@with_phil
def create(ObjectID loc not None, uint64_t trans_num):
    """(ObjectID loc, uint64_t trans_num) => TransactionID

    Create a new transaction.
    """
    return TransactionID(H5TRcreate(loc.id, trans_num))


cdef class TransactionID(ObjectID):

    """ HDF5 transaction object identifier class """

    @with_phil
    def get_trans_num(self):
        """() => UINT transaction_number

        Get the transaction number associated with the transaction.
        """
        cdef uint64_t trans_num
        H5TRget_trans_num(self.id, &trans_num)
        return trans_num


    @with_phil
    def commit(self):
        """()

        Commit the transaction.
        """
        H5TRcommit(self.id)
