#!/usr/bin/env python

# Make sure correct h5py module is imported...
import os
import sys
curr_dir = os.path.abspath(os.path.dirname(__file__))
h5py_dir = os.path.abspath(os.path.join(curr_dir, os.path.pardir,
                                        "build", "lib.linux-x86_64-2.6"))
if not os.path.isdir(h5py_dir):
    raise RuntimeError('%s: Not a directory' % h5py_dir)
sys.path.insert(1, h5py_dir)
#print 'sys.path = ', sys.path

from mpi4py import MPI
from h5py.eff_control import eff_init, eff_finalize
from h5py import h5, h5p, File, Dataset
import numpy as np

comm = MPI.COMM_WORLD
eff_init(comm, MPI.INFO_NULL)
my_rank = comm.Get_rank()
my_size = comm.Get_size()
print 'size = %d; rank = %d' % (my_size, my_rank)
f = File(os.environ["USER"]+"_ff_file_ex1.h5", mode='w', driver='iod',
         comm=comm, info=MPI.INFO_NULL)
my_version = 1
version = f.acquire_context(my_version)
print "Requested read context version = %d" % my_version
print "Acquired read context version = %d" % version

comm.Barrier()

##############

if my_rank == 0:
    f.create_transaction(2)
    f.tr.start()

    dset = f.create_dataset("/foo/bar/baz", shape=(10, 10), dtype='<i4')
    assert isinstance(dset, Dataset)
    assert "/foo/bar/baz" in f

    f.tr.finish()
#    f.tr.close()


f.rc.release()

comm.Barrier()

if my_rank == 0:
    dset.close()
    # grp2.close()

##############
#f.rc.close()
f.close()
#es.close()
#comm.Barrier()
eff_finalize()

print 'The End'
