# Testing HDF5 H5V Exascale FastForward

import numpy as np
from .common_ff import ut, TestCaseFF
from h5py.highlevel import File, View, AQuery, CQuery
from h5py.eff_control import eff_init, eff_finalize
# from h5py import h5t, h5a



class BaseTest(TestCaseFF):

    def setUp(self):
        self.ff_cleanup()
        self.start_h5ff_server(quiet=False)
        self.fname = self.filename("ff_file_view.h5")


    def tearDown(self):
        self.shut_h5ff_server()



class TestView(BaseTest):

    """
        Feature: View (H5V) operations
    """

    # @ut.skip('Test works')
    def test_view_001(self):
        """ View creation on file, group, dataset, attribute"""
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
            g = f.create_group('G')
            d = g.create_dataset('D', (10,))

            f.tr.finish()
            f.rc.release()
            f.acquire_context(2)

            q = AQuery('link_name') == 'foo'
            v1 = View(f, q)
            self.assertIsInstance(v1, View)
            v2 = View(g, q)
            v3 = View(d, q)
            with self.assertRaises(TypeError):
                v4 = View(f.attrs['a'], q)

            f.rc.release()

            d.close()
            g.close()
            f.close()
        eff_finalize()
