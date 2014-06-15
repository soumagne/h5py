# Testing HDF5 Exascale FastForward
# Tests adapted from the original h5py's collection of dataset tests

import sys
import numpy as np
from .common_ff import ut, TestCaseFF
from h5py.highlevel import File, Group, Dataset
from h5py import h5t
import h5py

if not h5.get_config().eff:
    raise RuntimeError('The h5py module was not built for Exascale FastForward')



class BaseTest(TestCaseFF):

    def setUp(self):
        self.ff_cleanup()
        self.start_h5ff_server()
        self.fname = self.filename("ff_file_dataset.h5")


    def tearDown(self):
        pass



class TestDataset(BaseTest):

    """
        Tests for HDF5 Exascale FastForward Dataset object
    """

    def test_create_scalar(self):
        """ Create a scalar dataset """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        rank = comm.Get_rank()
        f = File(self.fname, 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        f.acquire_context(1)
        comm.Barrier()
        if rank == 0:
            f.create_transaction(2)
            f.tr.start()

            dset = f.create_dataset('foo', ())
            self.assertIsInstance(dset, Dataset)
            self.assertEqual(dset.shape, ())

            f.tr.finish()
        f.rc.release()
        comm.Barrier()
        if rank == 0:
            dset.close()
        f.close()
        eff_finalize()


    def test_create_simple(self):
        """ Create a size-1 dataset in a group and confirm default dtype """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        rank = comm.Get_rank()
        f = File(self.fname, 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        f.acquire_context(1)
        comm.Barrier()
        if rank == 0:
            f.create_transaction(2)
            f.tr.start()

            g = f.create_group('G')
            dset = g.create_dataset('D', (63,))

            self.assertIsInstance(dset, Dataset)
            self.assertEqual(dset.shape, (63,))
            self.assertEqual(dset.dtype, np.dtype('=f4'))

            self.assertIsNone(f.ctn)
            self.assertEqual(str(g.ctn), str(f.ctn))
            self.assertEqual(str(dset.ctn), str(g.ctn))

            self.assertEqual(str(g.tr), str(f.tr))
            self.assertEqual(str(dset.tr), str(g.tr))

            self.assertEqual(str(g.rc), str(f.rc))
            self.assertEqual(str(dset.rc), str(g.rc))

            self.assertEqual(str(g.es), str(f.es))
            self.assertEqual(str(dset.es), str(g.es))

            f.tr.finish()
        f.rc.release()
        comm.Barrier()
        if rank == 0:
            dset.close()
            g.close()
        f.close()
        eff_finalize()


    def test_create_extended(self):
        """ Create an extended dataset and missing shape raises TypeError """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        rank = comm.Get_rank()
        f = File(self.fname, 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        f.acquire_context(1)
        comm.Barrier()
        if rank == 0:
            f.create_transaction(2)
            f.tr.start()

            dset = f.create_dataset('foo', (63,))
            self.assertEqual(dset.shape, (63,))
            self.assertEqual(dset.size, 63)
            dset = f.create_dataset('bar', (6, 10))
            self.assertEqual(dset.shape, (6, 10))
            self.assertEqual(dset.size, (60))

            with self.assertRaises(TypeError):
                f.create_dataset('baz')

            f.tr.finish()
        f.rc.release()
        comm.Barrier()
        if rank == 0:
            dset.close()
        f.close()
        eff_finalize()


    def test_create_scalar_extended(self):
        """ Create scalar and extended datasets from existing data """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        rank = comm.Get_rank()
        f = File(self.fname, 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        f.acquire_context(1)
        comm.Barrier()
        if rank == 0:
            f.create_transaction(2)
            f.tr.start()

            data = np.ones((), 'f')
            d1 = f.create_dataset('d1', data=data)
            self.assertEqual(d1.shape, data.shape)

            data = np.ones((100,), 'f')
            d2 = f.create_dataset('d2', data=data)
            self.assertEqual(d2.shape, data.shape)
            
            f.tr.finish()
        f.rc.release()
        comm.Barrier()
        if rank == 0:
            d1.close()
            d2.close()
        f.close()
        eff_finalize()


    def test_dataset_intermediate_group(self):
        """ Create dataset with missing intermediate groups """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        rank = comm.Get_rank()
        f = File(self.fname, 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        f.acquire_context(1)
        comm.Barrier()
        if rank == 0:
            f.create_transaction(2)
            f.tr.start()

            ds = f.create_dataset("/foo/bar/baz", shape=(10, 10), dtype='<i4')
            self.assertIsInstance(ds, h5py.Dataset)
            self.assertTrue("/foo/bar/baz" in self.f)
            
            f.tr.finish()
        f.rc.release()
        comm.Barrier()
        if rank == 0:
            d1.close()
            d2.close()
        f.close()
        eff_finalize()        
