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
        self.start_h5ff_server(quiet=False)
        self.fname = self.filename("ff_file_attrs.h5")


    def tearDown(self):
        pass



class TestAccess(BaseTest):

    """
        Feature: Attribute creation/retrieval via special methods
    """

    @ut.skip('Test works')
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

            #File "h5a.pyx", line 432, in h5py.h5a.iterate (h5py/h5a.c:5695)
            # H5Aiterate2(loc.id, <H5_index_t>index_type, <H5_iter_order_t>order,
            # RuntimeError: link iteration failed (Symbol table: Iteration failed)
            # self.assertEqual(f.attrs.keys(), ['a'])
            self.assertEqual(len(f.attrs), 1)
            self.assertEqual(f.attrs['a'], 4.0)

            with self.assertRaises(KeyError):
                f.attrs['b']

            f.rc.release()

            f.close()
        eff_finalize()


    @ut.skip('Test works')
    def test_overwrite(self):
        """ Attributes are silently overwritten """
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
            f.attrs['a'] = 5.0

            f.tr.finish()
            f.rc.release()
            f.acquire_context(2)

            self.assertEqual(f.attrs['a'], 5.0)

            f.rc.release()

            f.close()
        eff_finalize()


    # @ut.skip('Test works')
    def test_rank(self):
        """ Attribute rank is preserved """
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

            f.attrs['a'] = (4.0, 5.0)
            f.attrs['b'] = np.ones((1,))

            f.tr.finish()
            f.rc.release()
            f.acquire_context(2)

            self.assertEqual(f.attrs['a'].shape, (2,))
            self.assertArrayEqual(f.attrs['a'], np.array((4.0,5.0)))

            out = f.attrs['b']
            self.assertEqual(out.shape, (1,))
            self.assertEqual(out[()], 1)

            f.rc.release()

            f.close()
        eff_finalize()
