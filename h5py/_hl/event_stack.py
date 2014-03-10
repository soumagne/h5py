# This file implements the high-level Python interface to the HDF5 Event Stack
# API

from h5py import h5, version, h5es

# Check if this module can be used
mpi = h5.get_config().mpi
eff = h5.get_config().eff
if not (mpi and eff):
    raise RuntimeError('This module requires h5py to be built with MPI and HDF5 '
                       'FastForward library')

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
        """Event stack ID object"""
        return self._id


    def close(self):
        """Closes an event stack.
        
        At this point the event stack identifier becomes invalid.
        """
        self._id._close()
