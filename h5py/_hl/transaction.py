"""
Python high-level interface for the H5TR API
"""

from h5py import h5, version, h5tr

# Check if this module can be used
mpi = h5.get_config().mpi
eff = h5.get_config().eff
if not (mpi and eff):
    raise RuntimeError('This module requires h5py to be built with MPI and HDF5 '
                       'FastForward library')

class Transaction:
    """
    Represents an HDF5 FastForward transaction.
    """

    def __init__(self, f, rc, tnum):
        self._id = h5tr.create(f.id, rc.id, tnum)


    @property
    def id(self):
        """ Returns transaction ID"""
        return self._id


    def close(self):
        self._id._close()
