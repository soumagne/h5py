"""
Python interface to Exascale FastForward HDF5 Index (H5X) API
"""

from h5py import h5x

_indexes = {'dummy': h5x.PLUGIN_DUMMY,
            'fastbit': h5x.PLUGIN_FASTBIT,
            'alacrity': h5x.PLUGIN_ALACRITY}

class Index(object):
    """
    Mixin class for adding access to the H5X indexing HDF5_FF API.

    For Exascale FastForward.
    """

    def create_index(self, tr, plugin='dummy', esid=None):
        """Create an index of type plugin on this object.

        Arguments:

        tr
            Transaction object.

        plugin
            Plugin type. Possible values: 'dummy', 'fastbit', or 'alacrity'.
            Default 'dummy'.

        esid
            Optional EventStackID object. Default None.
        """
        plugin_id = _indexes[plugin]
        h5x.create_ff(self.container.id, plugin_id, self.id, es=esid)


    def remove_index(self, tr, plugin='dummy', es=None):
        """Remove an index of type plugin on this object.

        Arguments:

        tr
            Transaction object.

        plugin
            Plugin type. Possible values: 'dummy', 'fastbit', or 'alacrity'.
            Default 'dummy'.

        esid
            Optional EventStackID object. Default None.
        """
        plugin_id = _indexes[plugin]
        h5x.remove_ff(self.container.id, plugin_id, self.id, tr.id, es=esid)


    def index_count(self, rc, esid=None):
        """Number of indexes on this object.

        Arguments:

        rc
            Read context object.

        esid
            Optional EventStackID object. Default None.
        """
        count = h5x.get_count_ff(self.id, rc.id, es=esid)
        return count
