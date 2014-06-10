from .common import ut, TestCase
import os
import subprocess
import time


class TestCaseFF(TestCase):
    """
    Testing class for the Exascale FastForward project.
    """

    @property
    def num_ions(self):
        """Number of h5ff_server processes."""
        if self._num_ions is None:
            raise ValueError("Number of ION processes unknown")
        else:
            return self._num_ions


    @property
    def eff_mpi_ions(self):
        """String of Lola ION hostnames delimited by comma."""
        if "EFF_MPI_IONS" not in os.environ \
                or os.environ["EFF_MPI_IONS"] == '':
            raise ValueError("$EFF_MPI_IONS not set")
        ions = os.environ["EFF_MPI_IONS"]
        if ions[-1] == ',':
            ions = [:-1]
        return ions


    @property
    def eff_mpi_cns(self):
        """String of Lola CN hostnames delimited by comma."""
        if "EFF_MPI_CNS" not in os.environ \
                or os.environ["EFF_MPI_CNS"] == '':
            raise ValueError("$EFF_MPI_CNS not set")
        cns = os.environ["EFF_MPI_CNS"]
        if cns[-1] == ',':
            cns = [:-1]
        return cns


    @property
    def h5ff_server(self):
        """Full path to the h5ff_server executable."""
        if "H5FF_SERVER" not in os.environ:
            raise ValueError("$H5FF_SERVER not set")
        h5ff_server = os.environ["H5FF_SERVER"]
        if not os.path.isfile(h5ff_server):
            raise RuntimeError("%s: Not a file" % h5ff_server)
        elif not os.access(h5ff_server, os.X_OK):
            raise RuntimeError("%s: Not executable" % h5ff_server)
        return h5ff_server


    def ff_cleanup(self):
        """Run the FF cleanup shell script."""
        sname = r'/scratch/iod/scripts/purgeall.sh > /dev/null 2>&1'
        try:
            os.system(sname)
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError('%s: Failed to run' % sname)

        try:
            os.remove("./port.cfg")
        except:
            pass


    def filename(fname):
        """Produce a file name compatible with the Lola cluster's rules."""
        user = os.environ.get("USER", '')
        if len(user) == 0:
            raise ValueError("$USER env. varianble not set")
        return "%s_%s" % (user, fname)


    def start_h5ff_server(self, num_ions=1, sleep=2):
        """Start the h5ff_server and return its subprocess object.

        Arguments:

        num_ions
            Number of server processes (mpiexec's -np option).

        sleep
            Seconds to sleep after starting the server. Does not have to be an
            integer number.
        """
        num_ions = int(num_ions)
        self._num_ions = num_ions
        if num_ions > len(self.eff_mpi_ions.split(',')):
            raise ValueError("%d more IONs requested than available" % num_ions)

        cmd = ["mpiexec", "-np", str(num_ions), "-hosts", self.eff_mpi_ions,
               self.h5ff_server]
        cmdline = ' '.join(cmd)
        servp = subprocess.Popen(cmdline, shell=True)
        time.sleep(sleep)
        retcode = servp.poll()
        if retcode is not None:
            if retcode != 0:
                raise RuntimeError("%s: Failed to start h5ff_server (code: %d)"
                                   % (cmdline, retcode))
            else:
                raise RuntimeError("h5ff_server start process finished early")
        return servp


    #def shut_h5ff_server(self):
        #""" Shuts down all h5ff_server processes """
        #retcode = os.system("killall -g h5ff_server")
        #if retcode:
            #raise RuntimeError('killall -g h5ff_server: Command failed')


    def run_demo(self, fname, np=1):
        """Runs test/demo in file fname. fname is assumed to be in the tests
        directory.

        Arguments:

        fname
            Test's file name.

        np
            Number of processes to run the test.
        """
        # Figure out the h5py's import directory...
        curr_dir = os.path.abspath(os.path.dirname(__file__))
        h5py_dir = os.path.abspath(os.path.join(curr_dir, os.path.pardir,
                                                os.path.pardir, os.path.pardir))
        if not os.path.isdir(h5py_dir):
            raise RuntimeError('%s: Not a directory' % h5py_dir)

        # Command line to execute...
        np = int(np)
        cmd = ["mpiexec", "-np", str(np), "-hosts", self.eff_mpi_cns,
               os.path.join(curr_dir, fname), h5py_dir]
        cmdline = ' '.join(cmd)

        # Start h5ff_server processes...
        servp = self.start_h5ff_server(num_ions=np)

        # Run the command and wait for it to finish...
        proc = subprocess.Popen(cmdline, shell=True)
        proc.wait()

        if servp.poll() is None:
            servp.terminate()

        return proc.returncode
