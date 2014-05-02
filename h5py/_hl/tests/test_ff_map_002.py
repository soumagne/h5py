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
        self.shut_h5ff_server()
        self.ff_cleanup()
        os.chdir(self._old_dir)


class TestMap(BaseTest):

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
            print ">>>>> second set()"
            m.set('b', 2, f.tr)

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

            print ">>>>> get('a')"
            val = m.get('a', f.rc)
            print ">>>>> key('a') =", val

            m.close()

        f.rc.release()

        comm.Barrier()

        f.rc._close()

        f.close()
        es.close()
        eff_finalize()

