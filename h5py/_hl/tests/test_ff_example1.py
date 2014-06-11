# Test suite for Exascale FastForward "Example1".

import os
from .common_ff import TestCaseFF
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


class BaseTest(TestCaseFF):
    
    def setUp(self):
        self.ff_cleanup()
        self.start_h5ff_server()


    def tearDown(self):
        #self.shut_h5ff_server()
        self.ff_cleanup()


class TestExample1(BaseTest):

    def test_example1(self):
        """File create/close"""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        es = EventStack()
        fname = self.filename("ff_file_ex1.h5")
        f = File(fname, es, mode='w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        f.es = es
        f.close()
        eff_finalize()


    def test_example2(self):
        """Group create in a file"""
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()

        es = EventStack()
        self.assertIsInstance(es, EventStack)
        
        fname = self.filename("ff_file_ex1.h5")
        f = File(fname, es, mode='w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        f.es = es
        self.assertIsInstance(f.es, EventStack)
        
        my_version = 1
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)

        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(2)
            f.tr.start()

            grp1 = f.create_group("G1")
            grp2 = grp1.create_group("G2")

            f.tr.finish()
        
        f.rc.release()
        
        comm.Barrier()
        
        if my_rank == 0:
            grp1.close()
            grp2.close()

        f.close()
        
        eff_finalize()
