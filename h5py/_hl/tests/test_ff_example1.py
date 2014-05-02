# Test suite for Exascale FastForward "Example1".

import os
from .common_ff import TestCase_ff
from h5py import h5
from h5py.eff_control import eff_init, eff_finalize

from h5py.highlevel import EventStack, File

# Check if this HDF5 is built with MPI and for EFF...
mpi = h5.get_config().mpi
if not mpi:
    raise RuntimeError('This HDF5 does not appear to be built with MPI')
eff = h5.get_config().eff
if not eff:
    raise RuntimeError('The h5py module was not built for Exascale FastForward') 


class BaseTest(TestCase_ff):
    
    def setUp(self):
        self.ff_cleanup()
        self._old_dir = os.getcwd()
        os.chdir(self.exe_dir)
        self.run_h5ff_server()


    def tearDown(self):
        self.shut_h5ff_server()
        self.ff_cleanup()
        os.chdir(self._old_dir)


class TestExample1(BaseTest):

    def test_example1(self):
        """ Example 1 """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        es = EventStack()
        f = File('ff_file_ex1.h5', 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        f.close()
        eff_finalize()


    def test_example2(self):
        """ Example 2 """
        from mpi4py import MPI
        from h5py import h5p

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_ex1.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start(h5p.DEFAULT)

            grp1 = f.create_group("G1", f.tr)
            grp2 = grp1.create_group("G2", f.tr)

            f.tr.finish()
            # f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        if my_rank == 0:
            grp1.close()
            grp2.close()
        #f.rc._close()
        f.close()
        #es.close()
        #comm.Barrier()
        eff_finalize()
