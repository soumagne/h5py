#!/usr/bin/env python

# Make sure that HDF5_FF library is found...
import sys
sys.path.insert(1, '/home/ajelenak/h5py/build/lib.linux-x86_64-2.6')
#print 'sys.path = ', sys.path

from mpi4py import rc
rc.initialize = False

from mpi4py import MPI
from h5py import h5, File, EventStack
from h5py.eff_control import eff_init as EFF_init, eff_finalize as EFF_finalize

print "mpi = ", h5.get_config().mpi
print "HDF5 version = ", h5.get_libversion()

provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
print 'MPI.THREAD_MULTIPLE = %d' % MPI.THREAD_MULTIPLE
print 'provided = %d' % provided

comm = MPI.COMM_WORLD
EFF_init(comm, MPI.INFO_NULL)
print 'size = %d; rank = %d' % (comm.Get_size(), comm.Get_rank())
es = EventStack()
f = File_ff('ff_file_ex1.h5', 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)
f.close()
EFF_finalize()

print 'The End'
