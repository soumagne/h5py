"""
Exascale FastForward initialization and finalization functions
"""

include "config.pxi"

IF MPI and HDF5_VERSION >= (1, 9, 170):
    from mpi4py import MPI.Comm as Comm, MPI.Info as Info, MPI.INFO_NULL as INFO_NULL

    def eff_init(Comm comm not None, Info info):
        """(Comm comm, Info info)

        Initialize the Exascale FastForward stack.

        Comm: An mpi4py.MPI.Comm instance
        Info: An mpi4py.MPI.Info instance
        """

        if info is None:
            info = INFO_NULL
        EFF_Init(comm.ob_mpi, info.ob_mpi)


    def eff_finalize():
        """()

        Shutdown the Exascale FastForward stack.
        """
        EFF_finalize()
