#!/usr/bin/env python

# Make sure the correct h5py module will be imported...
import sys

print "h5py import dir =", sys.argv[1]
sys.path.insert(1, sys.argv[1])

from h5py.eff_control import eff_init, eff_finalize
from h5py.highlevel import EventStack, File, Map
from h5py.highlevel import Datatype
from mpi4py import MPI
import numpy

print ">>>>> in ff_map_kv_ops_user_types"

comm = MPI.COMM_WORLD
eff_init(comm, MPI.INFO_NULL)
my_rank = comm.Get_rank()
es = EventStack()
f = File('ff_file_map.h5', 'w', driver='iod', comm=comm,
         info=MPI.INFO_NULL)
my_version = 0
version = f.acquire_context(my_version)
assert my_version == version, "Read context version: %d, requested %d" \
        % (version, my_version)

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
assert my_version == version, "Read context version: %d, requested %d" \
        % (version, my_version)

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
