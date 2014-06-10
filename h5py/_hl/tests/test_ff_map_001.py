# Test suite for Exascale FastForward H5M API.

import os
from .common_ff import ut, TestCaseFF
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


class BaseTest(TestCaseFF):
    
    def setUp(self):
        self.ff_cleanup()
        self.start_h5ff_server()


    def tearDown(self):
        #self.shut_h5ff_server()
        self.ff_cleanup()


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
            kv_pairs = m.count(f.rc)
            self.assertEqual(kv_pairs, 0)
            with self.assertRaises(KeyError):
                val = m.get(1, f.rc)

        m.close()

        f.rc.release()

        comm.Barrier()

        f.rc._close()

        f.close()
        es.close()
        eff_finalize()
