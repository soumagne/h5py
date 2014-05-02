from .common import ut, TestCase
import os
import subprocess
import time


class TestCase_ff(TestCase):
    """
    Testing class for the Exascale FastForward project.
    """

    def ff_cleanup(self):
        """ Runs the FF clean up shell script. """

        sname = r'/home/ajelenak/bin/ff_cleanup.sh > /dev/null 2>&1'
        try:
            os.system(sname)
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError('%s: Failed to run' % sname)


    @property
    def exe_dir(self):
        """ Directory from which to run tests. """
        return r'/scr/ajelenak/hdf5ff_build/examples'


    def run_h5ff_server(self, sleep=0.2):
        """ Runs the h5ff_server executable. """

        cmd = ['/scr/chaarawi/install/mpich3/bin/mpiexec',
               '-np 1',
               self.exe_dir + '/h5ff_server &']
        cmdline = ' '.join(cmd)
        retcode = os.system(cmdline)
        if retcode:
            raise RuntimeError('h5ff_server: Failed to run')
        time.sleep(sleep)

    def shut_h5ff_server(self):
        """ Shuts down all h5ff_server processes """

        retcode = os.system('killall -g h5ff_server')
        if retcode:
            raise RuntimeError('killall -g h5ff_server: Command failed')
