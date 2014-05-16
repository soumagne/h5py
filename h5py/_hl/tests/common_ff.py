from .common import ut, TestCase
import os
import subprocess
import time


class TestCase_ff(TestCase):
    """
    Testing class for the Exascale FastForward project.
    """

    @property
    def exe_dir(self):
        """ Directory from which to run tests. """
        return r'/scr/ajelenak/build/examples'


    @property
    def h5ff_server(self):
        """ Name of the h5ff_server's executable """
        return r'h5ff_server'


    def ff_cleanup(self):
        """ Runs the FF clean-up shell script. """

        sname = r'/home/ajelenak/bin/ff_cleanup.sh > /dev/null 2>&1'
        try:
            os.system(sname)
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError('%s: Failed to run' % sname)


    def run_h5ff_server(self, sleep=0.2):
        """ Runs the h5ff_server executable. """

        cmd = ['/scr/chaarawi/install/mpich3/bin/mpiexec', '-np 1',
                self.exe_dir + '/' + self.h5ff_server, '&']
        cmdline = ' '.join(cmd)
        retcode = os.system(cmdline)
        if retcode:
            raise RuntimeError('h5ff_server: Failed to run')
        time.sleep(sleep)


    def shut_h5ff_server(self):
        """ Shuts down all h5ff_server processes """

        retcode = os.system('killall -g ' + self.h5ff_server)
        if retcode:
            raise RuntimeError('killall -g h5ff_server: Command failed')


    def run_demo(self, fname):
        """ Runs test/demo in fname. fname is assumed to be in the tests
        directory. """

        # Figure out the h5py's import directory...
        curr_dir = os.path.abspath(os.path.dirname(__file__))
        h5py_dir = os.path.abspath(os.path.join(curr_dir, os.path.pardir,
                                                os.path.pardir, os.path.pardir))
        if not os.path.isdir(h5py_dir):
            raise RuntimeError('%s: Not a directory' % h5py_dir)

        # Command line to execute...
        cmd = 'mpiexec -np 1 ./h5ff_server & sleep 1; mpirun -np 1 python '
        cmd += os.path.join(curr_dir, fname)
        cmd += ' ' + h5py_dir

        # Run the command and wait for it to finish...
        proc = subprocess.Popen(cmd, shell=True, cwd=self.exe_dir)
        proc.wait()

        return proc.returncode
