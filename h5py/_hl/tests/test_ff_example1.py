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
        os.chdir(self.exe_dir)
        self.run_h5ff_server()


    def tearDown(self):
        self.ff_cleanup()
        os.chdir(self._old_dir)


class TestMPI(BaseTest):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mpi_thread_multi(self):
        """ MPI_THREAD_MULTIPLE support """
        from mpi4py import MPI
        provided = MPI.Query_thread()
        self.assertEqual(provided, MPI.THREAD_MULTIPLE)


    def test_mpi_auto_init(self):
        """ MPI auto initialization """
        from mpi4py import MPI
        self.assertTrue(MPI.Is_initialized())


class TestExample1(BaseTest):

    def test_simple(self):
        """ Simple Example1 """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        es = EventStack()
        f = File('ff_file_ex1.h5', 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        f.close()
        eff_finalize()
