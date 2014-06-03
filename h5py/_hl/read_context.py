"""
Python high-level interface for the H5RC API
"""

from h5py import h5, h5rc

# Check if this module can be used
eff = h5.get_config().eff
if not eff:
    raise RuntimeError("This module requires h5py to be built with HDF5 "
                       "FastForward library")


_version = {"exact": h5rc.EXACT,
            "prev": h5rc.PREV,
            "next": h5rc.NEXT,
            "last": h5rc.LAST}


def make_rcapl(req="exact"):
    """Prepare a read context acquire property list based on input arguments.
    """
    rcapl = h5p.create(h5p.RC_ACQUIRE)
    rcapl.set_version_request(_version[req])
    return rcapl


class ReadContext(object):
    """
    Represents an HDF5 FastForward read context.
    """

    @property
    def id(self):
        """ Returns read context ID"""
        return self._id


    @property
    def version(self):
        """ Container version of this read context """
        return self._id.get_version()


    def __init__(self, rcid):
        """ Initialize this instance. Argument:

        rcid
            A RCntxtID object to wrap with this class.
        """
        if not isinstance(rcid, h5rc.RCntxtID):
            raise TypeError("%s is not h5rc.RCntxtID" % rcid)
        self._id = rcid


    def __repr__(self):
        if not self._id:
            return "<HDF5 closed read context>"
        else:
            v = self._id.get_version()
            return "<HDF5 read context at version %d (%s)>" \
                % (v, id(self._id))


    def close(self):
        """ Close the read context """
        self._id._close()


    def persist(self, es=None):
        """Copy data for the container from IOD to DAOS up to specified
        container version.
        """
        self._id.persist(es=es)


    def release(self, es=None):
        """Close the read context and release the read handle for the associated
        container version.
        """
        self._id.release(es=es)


    def make_snapshot(self, name, es=None):
        """Make a named snapshot of the container on DAOS.
        """
        self._id.snapshot(name, es=es)
