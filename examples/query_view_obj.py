import h5py
import numpy as np

# Create simple test file
myfile = h5py.h5f.create("test_view_obj.h5")
mytype = h5py.h5t.NATIVE_INT32
myspace = h5py.h5s.create_simple((100,))
mydataset = h5py.h5d.create(myfile, "pressure", mytype, myspace)
h5py.h5d.create(myfile, "temperature", mytype, myspace)
myarray = np.zeros((100,), dtype='int32')
mydataset.write(h5py.h5s.ALL, myspace, myarray, mytype)
myfile.close()

# Create query link_name = "Pressure"
myquery_type = h5py.h5q.TYPE_LINK_NAME
myquery_matchop = h5py.h5q.MATCH_EQUAL
myquery = h5py.h5q.create(myquery_type, myquery_matchop, "pressure")

# Open file and apply query
myfile = h5py.h5f.open("test.h5", h5py.h5f.ACC_RDONLY)
myview = myquery.apply(myfile)
if h5py.h5q.REF_OBJ == myview[1]:
  print "Object reference found"

# Get view back (HDF5 group)
mygroup_view = myview[0]

# Read object reference
myref_name = h5py.h5q.VIEW_REF_OBJ_NAME
myref_dataset = h5py.h5d.open(mygroup_view, myref_name)
mynpref_type = h5py.special_dtype(ref=h5py.Reference)
myref_space = h5py.h5s.create_simple((1,))
myref_array = np.empty((1,), dtype=mynpref_type)
myref_type = h5py.h5t.py_create(mynpref_type)
myref_dataset.read(h5py.h5s.ALL, myref_space, myref_array, myref_type)

# Dereference object reference and get name
myderef_dataset = h5py.h5r.dereference(myref_array[0], myfile)
myderef_dataset_name = h5py.h5i.get_name(myderef_dataset)
print "Objet %s matches query" % myderef_dataset_name


