#!/usr/bin/env python

# Make sure that HDF5_FF library is found...
import sys
sys.path.insert(1, '../build/lib.linux-x86_64-2.6')
#print 'sys.path = ', sys.path

from mpi4py import rc
rc.initialize = False

import os
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
f = File(os.environ["USER"]+"_ff_file_ex1.h5", 'w', driver='iod', comm=comm,
         info=MPI.INFO_NULL)

my_version = 0
version = f.acquire_context(my_version)
print "Requested read context version = %d" % my_version
print "Acquired read context version = %d" % version

comm.Barrier()

if my_rank == 0:
    f.create_transaction(1)
    f.tr.start(h5p.DEFAULT)

    grp1 = f.create_group("G1", f.tr)
    grp2 = grp1.create_group("G2", f.tr)

    f.tr.finish()
#    f.tr._close()


f.rc.release()

comm.Barrier()

if my_rank == 0:
    grp1.close()
    grp2.close()

#f.rc._close()
f.close()
#es.close()
#comm.Barrier()
EFF_finalize()

print 'The End'
