# Test suite for Exascale FastForward H5M API.

import os
from .common_ff import ut, TestCaseFF
from h5py import h5
from h5py.eff_control import eff_init, eff_finalize
from h5py.highlevel import EventStack, File, Map
import numpy

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
        self.start_h5ff_server()


    def tearDown(self):
        pass


class TestMap(BaseTest):

    def test_map_kv_ops_user_types(self):
        """ Map get/set/delete/exists key/value ops with user supplied datatypes
        """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        fname = self.filename("ff_file_map.h5")
        f = File(fname, 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
        my_version = 1
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)

        comm.Barrier()

        if my_rank == 0:
            f.create_transaction(2)
            f.tr.start()

            m = f.create_map('test_map', f.tr, key_dtype='S7', val_dtype='int64')
            m.set('a', 1, f.tr)
            m.set('b', 2, f.tr)
            m.set(1, 3, f.tr) # will convert key to string '1'
            m.set('12345678', 4, f.tr) # will clip key to 7 chars

            f.tr.finish()
            f.tr._close()

        f.rc.release()
        comm.Barrier()
        f.rc._close()

        my_version = 2
        version = f.acquire_context(2)        
        self.assertEqual(my_version, version)

        comm.Barrier()

        if my_rank == 0:
            kv_exists = m.exists('a', f.rc)
            self.assertTrue(kv_exists)

            kv_exists = m.exists('b', f.rc)
            self.assertTrue(kv_exists)

            kv_exists = m.exists('c', f.rc)
            self.assertFalse(kv_exists)

            kv_exists = m.exists('1', f.rc)
            self.assertTrue(kv_exists)

            kv_exists = m.exists(1, f.rc)
            self.assertTrue(kv_exists)

            kv_exists = m.exists('1234567', f.rc)
            self.assertTrue(kv_exists)

            cnt = m.count(f.rc)
            self.assertEqual(cnt, 4)

            # Close the map object so it can be opened for a get()
            m.close()
            m = f.open_map('test_map', f.rc)

            val = m.get('a', f.rc)
            self.assertEqual(val, 1)

            val = m.get('b', f.rc)
            self.assertEqual(val, 2)

            val = m.get('1', f.rc)
            self.assertEqual(val, 3)

            val = m.get(1, f.rc)
            self.assertEqual(val, 3)

            val = m.get('1234567', f.rc)
            self.assertEqual(val, 4)

            # Delete some key-value pairs...
            f.create_transaction(3)
            f.tr.start()
            m.delete('b', f.tr)
            m.delete('1234567', f.tr)
            f.tr.finish()
            f.tr._close()

        f.rc.release()
        comm.Barrier()
        f.rc._close()

        my_version = 3
        version = f.acquire_context(3)
        self.assertEqual(my_version, version)

        comm.Barrier()

        if my_rank == 0:
            kv_exists = m.exists('a', f.rc)
            self.assertTrue(kv_exists)

            kv_exists = m.exists('b', f.rc)
            self.assertFalse(kv_exists)

            kv_exists = m.exists('1', f.rc)
            self.assertTrue(kv_exists)

            kv_exists = m.exists('1234567', f.rc)
            self.assertFalse(kv_exists)

            cnt = m.count(f.rc)
            self.assertEqual(cnt, 2)


        m.close()

        f.rc.release()
        comm.Barrier()
        f.rc._close()

        f.close()
        eff_finalize()
