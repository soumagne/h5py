from .common import ut, TestCase
import os
import subprocess


class TestCase_ff(TestCase):
    """
    Testing class for the Exascale FastForward project.
    """

    def ff_cleanup(self):
        """ Runs the FF clean up shell script. """

        sname = r'/home/ajelenak/bin/ff_cleanup.sh'
        try:
            subprocess.call(sname)
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError('%s: Failed to run' % sname)
#        retcode = subprocess.call(sname)
#        if retcode:
#            raise RuntimeError('%s: Failed to run' % sname)


    def _exe_dir(self):
        """ Directory from which to run tests. """
        return r'/scr/ajelenak/hdf5ff_build/examples'


    def run_h5ff_server(self):
        """ Runs the h5ff_server executable. """

#        cmd = ['/scr/chaarawi/install/mpich3/bin/mpiexec/mpiexec',
#               '-np 1',
#               self._exe_dir() + '/h5ff_server']
#        try:
#            subprocess.check_call(cmd)
#        except (OSError, subprocess.CalledProcessError) as e:
#            raise RuntimeError('h5ff_server: Failed to run')

        cmd = ['set;',
               '/scr/chaarawi/install/mpich3/bin/mpiexec',
               '-np 1',
               self._exe_dir() + '/h5ff_server &']
        cmdline = ' '.join(cmd)
        print 'cmdline is: %s' % cmdline
        retcode = os.system(cmdline)
        if retcode:
            raise RuntimeError('h5ff_server: Failed to run')
