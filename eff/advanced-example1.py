#!/usr/bin/env python

# Make sure that HDF5_FF library is found...
import sys
sys.path.insert(1, '/home/ajelenak/h5py/build/lib.linux-x86_64-2.6')
#print 'sys.path = ', sys.path

from mpi4py import rc
rc.initialize = False

from mpi4py import MPI
from h5py import h5, h5p, File, EventStack
from h5py.eff_control import eff_init as EFF_init, eff_finalize as EFF_finalize

print "mpi = ", h5.get_config().mpi
print "HDF5 version = ", h5.get_libversion()

provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
print 'MPI.THREAD_MULTIPLE = %d' % MPI.THREAD_MULTIPLE
print 'provided = %d' % provided

comm = MPI.COMM_WORLD
EFF_init(comm, MPI.INFO_NULL)
my_rank = comm.Get_rank()
my_size = comm.Get_size()
print 'size = %d; rank = %d' % (my_size, my_rank)
es = EventStack()
f = File('ff_file_ex1.h5', 'w', driver='iod', comm=comm, info=MPI.INFO_NULL)

my_version = 0
version = f.acquire_context(my_version)
print "Requested read context version = %d" % my_version
print "Acquired read context version = %d" % version

comm.Barrier()

if my_rank == 0:
    f.create_transaction(1)
    f.tr.start(h5p.DEFAULT)

    grp1 = f.create_group("G1")
    grp2 = grp1.create_group("G2")

    f.tr.finish()

    # H5TRclose() skipped


f.rc.release()

comm.Barrier()

# H5Gclose_ff() skipped
# H5RCclose() skipped

f.close()
EFF_finalize()

print 'The End'
