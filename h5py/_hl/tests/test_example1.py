# Test suite for Exascale FastForward "Example1".

import os
from .common_ff import TestCase_ff
from h5py import h5
from h5py.eff_control import eff_init, eff_finalize

from h5py.highlevel import EventStack, File_ff

# Check if this HDF5 is built with MPI...
mpi = h5.get_config().mpi
if not mpi:
    raise RuntimeError('This HDF5 does not appear to be built with MPI')


class BaseTest(TestCase_ff):
    
    def setUp(self):
        self.ff_cleanup()
        self._old_dir = os.getcwd()
        print "old dir = %s" % self._old_dir
        os.chdir(self._exe_dir())
        self.run_h5ff_server()
        print "passed run_h5ff_server()"


    def tearDown(self):
        self.ff_cleanup()
        os.chdir(self._old_dir)


class TestSimple(BaseTest):

    def test_mpi_thread_multi(self):
        """ MPI_THREAD_MULTIPLE support """
        print "In test..."
        print "cwd = %s" % os.getcwd()
        # Prevent MPI from automatically initializing at import
        print "before import rc"
        from mpi4py import rc
        print "after import rc"
        rc.initialize = False
        print "after rc.initialize"

        print "before import MPI"
        from mpi4py import MPI
        print "after import MPI"
        provided = MPI.Init_thread(MPI.THREAD_MULTIPLE)
        print "after MPI.Init_thread()"
        self.assertEqual(provided, MPI.THREAD_MULTIPLE)
