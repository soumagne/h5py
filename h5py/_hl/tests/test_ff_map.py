# Test suite for Exascale FastForward H5M API.

import os
from .common_ff import ut, TestCase_ff
from h5py import h5
from h5py.eff_control import eff_init, eff_finalize

from h5py.highlevel import EventStack, File, Map

# Check if this HDF5 is built with MPI and for EFF...
eff = h5.get_config().eff
if not eff:
    raise RuntimeError('The h5py module was not built for Exascale FastForward') 
mpi = h5.get_config().mpi
if not mpi:
    raise RuntimeError('This HDF5 does not appear to be built with MPI support')


class BaseTest(TestCase_ff):
    
    def setUp(self):
        self.ff_cleanup()
        self._old_dir = os.getcwd()
        os.chdir(self.exe_dir)
        self.run_h5ff_server()


    def tearDown(self):
        self.ff_cleanup()
        os.chdir(self._old_dir)


class TestMap(BaseTest):

    def test_create_map_root(self):
        """ Create an empty map in the root group """
        from mpi4py import MPI
        from h5py import h5p, h5m

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            m = f.create_map('empty_map', f.tr)

            self.assertIsInstance(m, Map)
            self.assertIsInstance(m.id, h5m.MapID)

            m.close()

            f.tr.finish()
            # f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        #f.rc._close()
        f.close()
        #es.close()
        eff_finalize()


    def test_default_kv_types(self):
        """ Default key/value datatypes """
        from mpi4py import MPI
        from h5py import h5p
        from h5py.highlevel import Datatype
        import numpy

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            m = f.create_map('empty_map', f.tr)

            self.assertIsInstance(m.key_dtype, Datatype)
            self.assertIsInstance(m.val_dtype, Datatype)
            self.assertEqual(m.key_dtype.dtype, numpy.dtype('=f4'))
            self.assertEqual(m.val_dtype.dtype, numpy.dtype('=f4'))

            m.close()

            f.tr.finish()
            # f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        #f.rc._close()
        f.close()
        #es.close()
        eff_finalize()


    def test_create_map_group(self):
        """ Create an empty map in a group """
        from mpi4py import MPI
        from h5py import h5p, h5m

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            grp1 = f.create_group("G1", f.tr)
            grp2 = grp1.create_group("G2", f.tr)

            m = grp2.create_map('empty_map', f.tr)
            self.assertIsInstance(m, Map)
            self.assertIsInstance(m.id, h5m.MapID)
            m.close()

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


    def test_create_open_group_map(self):
        """ Create an empty map in a group then open it """
        from mpi4py import MPI
        from h5py import h5m

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()

        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            grp1 = f.create_group("G1", f.tr)
            grp2 = grp1.create_group("G2", f.tr)

            m = grp2.create_map('empty_map', f.tr)
            self.assertIsInstance(m, Map)
            self.assertIsInstance(m.id, h5m.MapID)
            m.close()

            grp2.close()
            grp1.close()

            f.tr.finish()
            f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        f.rc._close()

        my_version = 1
        version = f.acquire_context(1)
        self.assertEqual(my_version, version)

        comm.Barrier()

        if my_rank == 0:
            m = f.open_map('G1/G2/empty_map', f.rc)
            self.assertIsInstance(m, Map)
            self.assertIsInstance(m.id, h5m.MapID)
            m.close()

        f.rc.release()
        comm.Barrier()
        f.rc._close()

        f.close()
        es.close()
        eff_finalize()


    def test_get_kv_types(self):
        """ h5m.get_types_ff reported datatypes """
        from mpi4py import MPI
        from h5py import h5p
        from h5py.highlevel import Datatype
        import numpy

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            m = f.create_map('empty_map', f.tr)
            key_dt = m.key_type(f.rc)
            val_dt = m.value_type(f.rc)

            self.assertIsInstance(key_dt, Datatype)
            self.assertIsInstance(val_dt, Datatype)
            self.assertEqual(key_dt.dtype, numpy.dtype('=f4'))
            self.assertEqual(val_dt.dtype, numpy.dtype('=f4'))

            m.close()

            f.tr.finish()
            # f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        #f.rc._close()
        f.close()
        #es.close()
        eff_finalize()


    def test_user_kv_types(self):
        """ User supplied datatypes """
        from mpi4py import MPI
        from h5py import h5p
        from h5py.highlevel import Datatype
        import numpy

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            m = f.create_map('empty_map', f.tr, key_dtype='S7',
                             val_dtype='int64')
            key_dt = m.key_type(f.rc)
            val_dt = m.value_type(f.rc)

            self.assertIsInstance(key_dt, Datatype)
            self.assertIsInstance(val_dt, Datatype)
            self.assertEqual(key_dt.dtype, numpy.dtype('S7'))
            self.assertEqual(val_dt.dtype, numpy.dtype('int64'))

            m.close()

            f.tr.finish()
            # f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        #f.rc._close()
        f.close()
        #es.close()
        eff_finalize()


    def test_get_empty_map(self):
        """ Getting a key/value pair from empty map raises exception  """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            m = f.create_map('empty_map', f.tr)
            kv_pairs = m.count(f.rc)

            self.assertEqual(kv_pairs, 0)
            with self.assertRaises(KeyError):
                val = m.get(1, f.rc)

            m.close()

            f.tr.finish()
            # f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        #f.rc._close()
        f.close()
        #es.close()
        eff_finalize()


    @ut.skip("Still being worked on")
    def test_map_kv_ops(self):
        """ Map set/get/delete/exist key/value operations  """
        from mpi4py import MPI

        print ">>>>> in test_map_kv_ops"

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            m = f.create_map('test_map', f.tr)

            # Number of key/value pairs in empty map...
            # kv_pairs = m.count(f.rc)
            # self.assertEqual(kv_pairs, 0)

            print ">>>>> first set()"
            m.set(1, 2, f.tr)
            print ">>>>> second set()"
            m.set(3, 4, f.tr)
            # print ">>>>> third set()"
            # m.set(5, 6, f.tr)

            # self.assertEqual(m.count(f.rc), 1)
            # with self.assertRaises(KeyError):
            #     val = m.get(1, f.rc)

            m.close()

            f.tr.finish()
            # f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        #f.rc._close()
        f.close()
        #es.close()
        eff_finalize()


    def test_map_kv_ops_user_types(self):
        """ Map get/set/delete/exists key/value ops with user supplied datatypes """
        from mpi4py import MPI
        from h5py import h5p
        from h5py.highlevel import Datatype
        import numpy

        print ">>>>> in test_map_kv_ops_user_types"

        comm = MPI.COMM_WORLD
        eff_init(comm, MPI.INFO_NULL)
        my_rank = comm.Get_rank()
        es = EventStack()
        f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
                 info=MPI.INFO_NULL)
        my_version = 0
        version = f.acquire_context(my_version)
        self.assertEqual(my_version, version)
        
        comm.Barrier()
        
        if my_rank == 0:
            f.create_transaction(1)
            f.tr.start()

            m = f.create_map('test_map', f.tr, key_dtype='S7',
                             val_dtype='int64')
            print ">>>>> m.id =", m.id
            print ">>>>> m.id.id =", m.id.id
            print ">>>>> f.tr =", f.tr
            print ">>>>> f.tr.id =", f.tr.id
            print ">>>>> first set()"
            m.set('a', 1, f.tr)
            # print ">>>>> second set()"
            # m.set('b', 2, f.tr)

            # m.close()

            f.tr.finish()
            f.tr._close()
        
        f.rc.release()
        
        comm.Barrier()
        
        f.rc._close()

        my_version = 1
        version = f.acquire_context(1)        
        self.assertEqual(my_version, version)

        comm.Barrier()

        if my_rank == 0:
            print ">>>>> exists('a')"
            kv_exists = m.exists('a', f.rc)
            print ">>>>> kv_exists =", kv_exists
            print ">>>>> exists('b')"
            kv_exists = m.exists('b', f.rc)
            print ">>>>> kv_exists =", kv_exists
            print ">>>>> count()"
            cnt = m.count(f.rc)
            print ">>>>> kv pair count =", cnt

            # Close the map object so it can be opened for a get()
            m.close()
            m = f.open_map('test_map', f.rc)

            # print ">>>>> get('a')"
            # val = m.get('a', f.rc)
            # print ">>>>> key('a') =", val

            m.close()

        f.rc.release()

        comm.Barrier()

        f.rc._close()

        f.close()
        es.close()
        eff_finalize()
