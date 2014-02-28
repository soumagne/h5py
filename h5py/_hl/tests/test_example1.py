# Test suite for Exascale FastForward "Example1".

import os
from .common_ff import TestCase_ff
from h5py import h5
from h5py.eff_control import eff_init, eff_finalize

from h5py.highlevel import EventStack, File

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
        #self.run_h5ff_server()
        print "passed run_h5ff_server()"


    def tearDown(self):
        self.ff_cleanup()
        os.chdir(self._old_dir)
        print "back in = %s" % os.getcwd()


class TestSimple(BaseTest):

    def test_mpi_thread_multi(self):
        """ MPI_THREAD_MULTIPLE support """
        print "In test..."
        print "cwd = %s" % os.getcwd()

        print "before import MPI"
        from mpi4py import MPI
        print "after import MPI"
        print "Is initialized?", MPI.Is_initialized()
        print "before MPI.Init_thread()"
        provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
        print "after MPI.Init_thread()"
        self.assertEqual(provided, MPI.THREAD_MULTIPLE)
