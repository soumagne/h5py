"""
Python high-level interface for the H5TR API
"""

from h5py import h5, h5tr, h5p

# Check if this module can be used
eff = h5.get_config().eff
if not eff:
    raise RuntimeError("This module requires h5py to be built with HDF5 "
                       "FastForward library")


def make_tspl(ldrs=None):
    """Produce a transaction start property list based on the input
    arguments.
    """
    tspl = h5p.create(h5p.TR_START)
    if ldrs is not None:
        ldrs = int(ldrs)
        tspl.set_num_peers(ldrs)

    return tspl


class Transaction(object):
    """
    Represents an HDF5 FastForward transaction.
    """

    @property
    def id(self):
        """Hold h5tr.TransactionID object."""
        return self._id


    def __init__(self, f, rc, tnum):
        """Create transaction associated with a specified container, read
        context, and transaction mumber.

        Arguments:

        f
            Container (File) object.

        rc
            Read context object.

        tnum
            Transaction number.
        """
        self._id = h5tr.create(f.id, rc.id, tnum)


    def __repr__(self):
        if not self._id:
            return "<HDF5 closed transaction>"
        else:
            v = self._id.get_version()
            return "<HDF5 transaction at version %d (%s)>" \
                % (v, id(self._id))


    def close(self):
        """Close the transaction."""
        self._id._close()


    def start(self, ldrs=1, es=None):
        """Start the transaction.

        Arguments:

        ldrs
            Number of leader processes. Default is 1.

        es
            Event stack object. Default is None.
        """
        tspl = None
        if ldrs > 1:
            tspl = make_tspl(ldrs)
        self._id.start(tspl=tspl, esid=es)


    def finish(self, with_rc=False, es=None):
        """Finish the transaction. Return a read context object if requested.
        Arguments:

        with_rc
            If set to True, new read context will be acquired just after this
            transaction finishes. Default is False.

        es
            Event stack object. Default is None.
        """
        rc = self._id.finish(rcid_flag=with_rc, esid=es)
        if with_rc:
            return rc


    def set_dependency(num, es=None):
        """Register dependency of the transaction on a lower-numbered
        transaction. Arguments:

        num
            Transaction number.

        es
            Event stack object. Default None.
        """
        self._id.set_dependency(num, esid=es)


    def abort(self, es=None):
        """Abort the transaction. Event stack object is an optional argument
        with default value of None.
        """
        self._id.abort(esid=es)
