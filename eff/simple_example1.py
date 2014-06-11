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

from mpi4py import rc
rc.initialize = False

from mpi4py import MPI
from h5py import h5, File, EventStack
from h5py.eff_control import eff_init as EFF_init, eff_finalize as EFF_finalize

print "mpi = ", h5.get_config().mpi
print "HDF5 version = ", h5.get_libversion()

provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
print "MPI.THREAD_MULTIPLE = %d" % MPI.THREAD_MULTIPLE
print "provided = %d" % provided

comm = MPI.COMM_WORLD
EFF_init(comm, MPI.INFO_NULL)
print "size = %d; rank = %d" % (comm.Get_size(), comm.Get_rank())
es = EventStack()
f = File(os.environ["USER"]+"_ff_file_ex1.h5", es, mode='w', driver="iod",
         comm=comm, info=MPI.INFO_NULL)
f.es = es
f.close()
EFF_finalize()

print "The End"
