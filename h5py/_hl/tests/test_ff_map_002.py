# Test suite for Exascale FastForward H5M API.

import os
from .common_ff import ut, TestCaseFF
from h5py import h5

# Check if this HDF5 is built with MPI and for EFF...
eff = h5.get_config().eff
if not eff:
    raise RuntimeError('The h5py module was not built for Exascale FastForward') 
mpi = h5.get_config().mpi
if not mpi:
    raise RuntimeError('This HDF5 does not appear to be built with MPI support')


class BaseTest(TestCaseFF):

    def setUp(self):
        self.ff_cleanup()


    def tearDown(self):
        self.ff_cleanup()


class TestMap(BaseTest):

    def test_map_kv_ops_user_types(self):
        """ Map get/set/delete/exists key/value ops with user supplied datatypes
        """

        r = self.run_demo('ff_map_kv_ops_user_types.py')
        self.assertEqual(r, 0)
