"""
Exascale FastForward initialization and finalization functions
"""

include "config.pxi"

IF MPI and HDF5_VERSION >= (1, 9, 170):

    def eff_init(Comm comm not None, Info info):
        """(Comm comm, Info info)

        Initialize the Exascale FastForward stack.

        Comm: mpi4py.MPI.Comm instance
        Info: mpi4py.MPI.Info instance
        """
        EFF_init(comm.ob_mpi, info.ob_mpi)


    def eff_finalize():
        """()

        Shutdown the Exascale FastForward stack.
        """
        EFF_finalize()
