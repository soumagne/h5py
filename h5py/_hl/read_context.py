"""
Python high-level interface for the H5RC API
"""

from h5py import h5, version, h5rc

# Check if this module can be used
mpi = h5.get_config().mpi
hdf5_ver = version.hdf5_version_tuple[0:3]
if not mpi or hdf5_ver < (1, 9, 170):
    raise RuntimeError('This module requires h5py to be built with MPI and HDF5 '
                       'FastForward library')

class ReadContext:
    """
    Represents an HDF5 FastForward read context.
    """

    def __init__(self, f, container_version):
        self._id = h5rc.create(f.id, container_version)


    @property
    def id(self):
        """ Returns read context ID"""
        return self._id


    def close(self):
        self._id._close()
