# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    H5R API for object and region references.
"""

cdef extern from "hdf5.h":
    herr_t H5Rcreate(void *ref, H5R_type_t ref_type, ...) except *

# Pyrex compile-time imports
from _objects cimport ObjectID, pdefault
from h5p cimport PropID

from ._objects import phil, with_phil


# === Public constants and data structures ====================================

OBJECT = H5R_OBJECT
DATASET_REGION = H5R_DATASET_REGION
REGION = H5R_REGION
ATTR = H5R_ATTR


# === Reference API ===========================================================

@with_phil
def create(int ref_type, *args):
    """(INT ref_type, *args) => ReferenceObject ref

    Create a new reference. The value of ref_type detemines the kind
    of reference created:

    OBJECT
        Reference to an object in an HDF5 file.  Parameters "loc"
        and "name" identify the object.

    DATASET_REGION
        Reference to a dataset region.  Parameters "loc" and
        "name" identify the dataset; the selection on "space"
        identifies the region.
    """
    cdef hid_t loc_id, space_id
    cdef char* name
    cdef Reference ref

    loc = args[0]
    if not isinstance(loc, ObjectID):
        raise ValueError("Second argument must be ObjectID")
    loc_id = loc.id

    obj_name = args[1]
    if not isinstance(obj_name, str):
        raise ValueError("Third argument must be string")
    name = obj_name

    if ref_type == H5R_OBJECT:
        ref = Reference()
        H5Rcreate(&ref.ref, <H5R_type_t>ref_type, loc_id, name)
    elif ref_type == H5R_DATASET_REGION:
        space = args[2]
        if not isinstance(space, ObjectID): # work around segfault in HDF5
            raise ValueError("Dataspace required for region reference")
        space_id = space.id
        ref = DsetRegionReference()
        H5Rcreate(&ref.ref, <H5R_type_t>ref_type, loc_id, name, space_id)
    elif ref_type == H5R_REGION:
        ref = RegionReference()
    elif ref_type == H5R_ATTR:
        ref = AttributeReference()
    else:
        raise ValueError("Unknown reference typecode")

    return ref


@with_phil
def dereference(Reference ref not None, ObjectID id not None, PropID oapl=None):
    """(Reference ref, ObjectID id, PropID oapl=None) => ObjectID or None

    Open the object pointed to by the reference and return its
    identifier.  The file identifier (or the identifier for any object
    in the file) must also be provided.  Returns None if the reference
    is zero-filled.

    The reference may be either Reference or DsetRegionReference.
    """
    import h5i
    if not ref:
        return None
    return h5i.wrap_identifier(H5Rdereference2(id.id, pdefault(oapl), <H5R_type_t>ref.typecode, &ref.ref))


@with_phil
def get_region(Reference ref not None, ObjectID id not None):
    """(Reference ref, ObjectID id) => SpaceID or None

    Retrieve the dataspace selection pointed to by the reference.
    Returns a copy of the dataset's dataspace, with the appropriate
    elements selected.  The file identifier or the identifier of any
    object in the file (including the dataset itself) must also be
    provided.

    The reference object must be a DsetRegionReference.  If it is zero-filled,
    returns None.
    """
    import h5s
    if not ref.typecode in (H5R_DATASET_REGION, H5R_REGION) or not ref:
        return None
    return h5s.SpaceID(H5Rget_region(id.id, <H5R_type_t>ref.typecode, &ref.ref))


@with_phil
def get_obj_type(Reference ref not None, ObjectID id not None):
    """(Reference ref, ObjectID id) => INT obj_code or None

    Determine what type of object the reference points to.  The
    reference may be a Reference or DsetRegionReference.  The file
    identifier or the identifier of any object in the file must also
    be provided.

    The return value is one of:

    - h5o.H5O_TYPE_GROUP
    - h5o.H5O_TYPE_DATASET

    If the reference is zero-filled, returns None.
    """
    cdef H5O_type_t type
    if not ref:
        return None
    H5Rget_obj_type2(id.id, <H5R_type_t>ref.typecode, &ref.ref, &type)
    return <int>type


@with_phil
def get_name(Reference ref not None, ObjectID loc not None):
    """(Reference ref, ObjectID loc) => STRING name

    Determine the name of the object pointed to by this reference.
    Requires the HDF5 1.8 API.
    """
    cdef ssize_t namesize = 0
    cdef char* namebuf = NULL

    namesize = H5Rget_name(loc.id, <H5R_type_t>ref.typecode, &ref.ref, NULL, 0)
    if namesize > 0:
        namebuf = <char*>malloc(namesize+1)
        try:
            namesize = H5Rget_name(loc.id, <H5R_type_t>ref.typecode, &ref.ref, namebuf, namesize+1)
            return namebuf
        finally:
            free(namebuf)


cdef class Reference:

    """
        Opaque representation of an HDF5 reference.

        Objects of this class are created exclusively by the library and
        cannot be modified.  The read-only attribute "typecode" determines
        whether the reference is to an object in an HDF5 file (OBJECT)
        or a dataset region (DATASET_REGION).

        The object's truth value indicates whether it contains a nonzero
        reference.  This does not guarantee that is valid, but is useful
        for rejecting "background" elements in a dataset.
    """

    def __cinit__(self, *args, **kwds):
        self.typecode = H5R_OBJECT
        self.typesize = sizeof(hobj_ref_t)

    def __nonzero__(self):
        cdef int i
        for i from 0<=i<self.typesize:
            if (<unsigned char*>&self.ref)[i] != 0: return True
        return False

    def __repr__(self):
        return "<HDF5 object reference%s>" % ("" if self else " (null)")

cdef class DsetRegionReference(Reference):

    """
        Opaque representation of an HDF5 region reference.

        This is a subclass of Reference which exists mainly for programming
        convenience.
    """

    def __cinit__(self, *args, **kwds):
        self.typecode = H5R_DATASET_REGION
        self.typesize = sizeof(hdset_reg_ref_t)

    def __repr__(self):
        return "<HDF5 region reference%s>" % ("" if self else " (null")

cdef class RegionReference(Reference):

    """
        Opaque representation of an HDF5 region reference.

        This is a subclass of Reference which exists mainly for programming
        convenience.
    """

    def __cinit__(self, *args, **kwds):
        self.typecode = H5R_REGION
        self.typesize = sizeof(hreg_ref_t)

    def __repr__(self):
        return "<HDF5 region reference%s>" % ("" if self else " (null")

cdef class AttributeReference(Reference):

    """
        Opaque representation of an HDF5 attribute reference.

        This is a subclass of Reference which exists mainly for programming
        convenience.
    """

    def __cinit__(self, *args, **kwds):
        self.typecode = H5R_ATTR
        self.typesize = sizeof(hattr_ref_t)

    def __repr__(self):
        return "<HDF5 attribute reference%s>" % ("" if self else " (null")








