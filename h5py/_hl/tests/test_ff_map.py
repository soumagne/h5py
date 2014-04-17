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
            f.tr.start(h5p.DEFAULT)

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
            f.tr.start(h5p.DEFAULT)

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


    # @ut.skip("Not working")
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
            f.tr.start(h5p.DEFAULT)

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
