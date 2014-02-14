"""
    H5RC API low-level Cython bindings
"""

include "config.pxi"

from h5f cimport FileID
from h5es cimport EventStackID, esid_default
from h5py import _objects

# Read Context operations

def create(FileID fid not None, int container_version):
    """(FileID fid, int container_version) => RCntxtID

    Create a read context associated with a container and version.
    """
    return RCntxtID.open(fid.id, <uint64_t>container_version)


# Read Context ID implementation

cdef class RCntxtID(ObjectID):
    """
    Represents an HDF5 read context identifier
    """

    def __cinit__(self, id):
        self.locked = True


    def close(self):
        """()

        Close a read context associated with this identifier.
        """
       with _objects.registry.lock:
           self.locked = False
            H5RCclose(self.id)
            _objects.registry.cleanup()


    def get_version(self):
        """() => UINT container_version

        Retrieve the container version associated with this read context
        identifier.
        """

        cdef uint64_t container_version
        H5RCget_version(self.id, &container_version)
        return container_version


    def release(self, EventStackID es=None):
        """(EventStackID es=None)

        Close the read context and release a read handle for the
        associated container version.
        """

        H5RCrelease (self.id, esid_default(es))


