import re
import warnings
import os.path as op

class BadLineError(Exception):
    pass

class UnknownCodeError(Exception):
    pass


# The following files are used to talk to the HDF5 api:
#
# (1) hdf5.pxd:         HDF5 function signatures    (autogenerated)
# (2) hdf5_types.pxd:   HDF5 type definitions       (static)
# (3) defs.pxd:         HDF5 function proxy defs    (autogenerated)
# (4) defs.pyx:         HDF5 function proxies       (autogenerated)

function_pattern = r'(?P<mpi>(MPI)[ ]+)?(?P<version>([0-9]\.[0-9]+\.[0-9]+))?([ ]+)?(?P<code>(unsigned[ ]+)?[a-zA-Z_]+[a-zA-Z0-9_]*\**)[ ]+(?P<fname>[a-zA-Z_]+[a-zA-Z0-9_]*)[ ]*\((?P<sig>[a-zA-Z0-9_,* ]*)\)'
sig_pattern = r'(unsigned[ ]+)?(?:[a-zA-Z_]+[a-zA-Z0-9_]*\**)[ ]+[ *]*(?P<param>[a-zA-Z_]+[a-zA-Z0-9_]*)'

fp = re.compile(function_pattern)
sp = re.compile(sig_pattern)

raw_preamble = """\
include "config.pxi"
from api_types_hdf5 cimport *
from api_types_ext cimport *

"""

def_preamble = """\
include "config.pxi"

from api_types_hdf5 cimport *
from api_types_ext cimport *

"""

imp_preamble = """\
include "config.pxi"
from api_types_ext cimport *
from api_types_hdf5 cimport *

cimport _hdf5

from _errors cimport set_exception

include "_locks.pxi"

rlock = FastRLock()
"""

class FunctionCruncher2(object):

    def __init__(self, stub=False):
        self.stub = stub

    def run(self):

        # Function definitions file
        self.functions = open(op.join('h5py', 'api_functions.txt'), 'r')

        # Create output files
        self.raw_defs =     open(op.join('h5py', '_hdf5.pxd'), 'w')
        self.cython_def =   open(op.join('h5py', 'defs.pxd'), 'w')
        self.cython_imp =   open(op.join('h5py', 'defs.pyx'), 'w')

        self.raw_defs.write(raw_preamble)
        self.cython_def.write(def_preamble)
        self.cython_imp.write(imp_preamble)

        for line in self.functions:
            if not line or line[0] == '#' or line[0] == '\n':
                continue
            try:
                self.handle_line(line)
            except BadLineError:
                raise
                warnings.warn("Skipped <<%s>>" % line)

        self.functions.close()
        self.cython_imp.close()
        self.cython_def.close()
        self.raw_defs.close()

    def handle_line(self, line):
        """ Parse a function definition line and output the correct code
        to each of the output files. """

        if line.startswith(' '):
            line = line.strip()
            if line.startswith('#'):
                return

            m = fp.match(line)
            if m is None:
                raise BadLineError(
                    "Signature for line <<%s>> did not match regexp" % line
                    )
            function_parts = m.groupdict()
            if function_parts['mpi'] is not None:
                function_parts['mpi'] = True
            if function_parts['version'] is not None:
                function_parts['version'] = tuple(int(x) for x in function_parts['version'].split('.'))

            self.raw_defs.write('  '+self.make_raw_sig(function_parts))
            self.cython_def.write(self.make_cython_sig(function_parts))
            self.cython_imp.write(self.make_cython_imp(function_parts))
        else:
            inc = line.split(':')[0]
            self.raw_defs.write('cdef extern from "%s.h":\n' % inc)

    def add_cython_if(self, function_parts, block):
        """ Wrap a block of code in the required "IF" checks """
        def wrapif(condition, code):
            code = code.replace('\n', '\n    ', code.count('\n')-1) # Yes, -1.
            code = "IF %s:\n    %s" % (condition, code)
            return code

        if function_parts['mpi']:
            block = wrapif('MPI', block)
        if function_parts['version']:
            block = wrapif('HDF5_VERSION >= %s' % (function_parts['version'],), block)

        return block

    def make_raw_sig(self, function_parts):
        """ Build a "cdef extern"-style definition for an HDF5 function """

        raw_sig = "%(code)s %(fname)s(%(sig)s) except *\n" % function_parts
        raw_sig = self.add_cython_if(function_parts, raw_sig)
        return raw_sig

    def make_cython_sig(self, function_parts):
        """ Build Cython signature for wrapper function """

        cython_sig = "cdef %(code)s %(fname)s(%(sig)s) except *\n" % function_parts
        cython_sig = self.add_cython_if(function_parts, cython_sig)
        return cython_sig

    def make_cython_imp(self, function_parts, stub=False):
        """ Build a Cython wrapper implementation. If stub is True, do
        nothing but call the function and return its value """

        args = sp.findall(function_parts['sig'])
        if args is None:
            raise BadLineError("Can't understand function signature <<%s>>" % function_parts['sig'])
        args = ", ".join(x[1] for x in args)

        # Figure out what conditional to use for the error testing
        code = function_parts['code']
        if '*' in code or code in ('H5T_conv_t',):
            condition = "==NULL"
            retval = "NULL"
        elif code in ('int', 'herr_t', 'htri_t', 'hid_t','hssize_t','ssize_t') \
          or re.match(r'H5[A-Z]+_[a-zA-Z_]+_t',code):
            condition = "<0"
            retval = "-1"
        elif code in ('unsigned int','haddr_t','hsize_t','size_t'):
            condition = "==0"
            retval = 0
        else:
            raise UnknownCodeError("Return code <<%s>> unknown" % self.code)

        parts = function_parts.copy()
        parts.update({'condition': condition, 'retval': retval, 'args': args})

        # Have to use except * because Cython can't handle special types here
        imp = """\
cdef %(code)s %(fname)s(%(sig)s) except *:
    cdef %(code)s r
    with rlock:
        r = _hdf5.%(fname)s(%(args)s)
        if r%(condition)s:
            if set_exception():
                return <%(code)s>%(retval)s;
        return r

"""

        stub_imp = """\
cdef %(code)s %(fname)s(%(sig)s) except *:
    with rlock:
        return hdf5.%(fname)s(%(args)s)

"""
        imp = (stub_imp if self.stub else imp) % parts
        imp = self.add_cython_if(function_parts, imp)
        return imp


def run(stub=False):
    fc = FunctionCruncher2(stub)
    fc.run()

if __name__ == '__main__':
    run()
