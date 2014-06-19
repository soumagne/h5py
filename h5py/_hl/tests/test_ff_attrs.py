# Testing HDF5 Exascale FastForward
# Tests adapted from the original h5py's collection of attribute tests

import numpy as np
from .common_ff import ut, TestCaseFF
from h5py.highlevel import File, AttributeManager
from h5py.eff_control import eff_init, eff_finalize
from h5py import h5t, h5a, get_config

if not get_config().eff:
    raise RuntimeError('The h5py module was not built for Exascale FastForward')



class BaseTest(TestCaseFF):

    def setUp(self):
        self.ff_cleanup()
        self.start_h5ff_server()
        self.fname = self.filename("ff_file_dataset.h5")


    def tearDown(self):
        pass



class TestAccess(BaseTest):

    """
        Feature: Attribute creation/retrieval via special methods
    """

    #@ut.skip('Test works')
    def test_create_scalar(self):
        """ Attribute creation by direct asignment """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        rank = comm.Get_rank()
        if rank == 0:
            f = File(self.fname, 'w', driver='iod', comm=comm,
                     info=MPI.INFO_NULL)
            f.acquire_context(1)
            f.create_transaction(2)
            f.tr.start()

            f.attrs['a'] = 4.0

            f.tr.finish()
            f.rc.release()
            f.acquire_context(2)

            self.assertEqual(self.f.attrs.keys(), ['a'])
            self.assertEqual(self.f.attrs['a'], 4.0)

            f.rc.release()

            f.close()
        eff_finalize()
