# This file implements the high-level Python interface to the HDF5 Event Stack
# API

from h5py import h5, version, h5es

# Currently does not work because mpi is false and version reported is 1.8.12.
# Check if this module can be used
#mpi = h5.get_config().mpi
#hdf5_ver = version.hdf5_version_tuple[0:3]
#if not mpi or hdf5_ver < (1, 9, 170):
#    raise RuntimeError('This module requires h5py to be built with MPI and HDF5 '
#                       'FastForward library')

class EventStack:
    """
    Represents an HDF5 event stack.

    Requires HDF5 FastForward (prereq.: MPI and Parallel HDF5).
    """

    def __init__(self):
        """
        Represents an event stack.
        """
        self._id = h5es.create()


    @property
    def id(self):
        """Event stack ID"""
        return self._id
