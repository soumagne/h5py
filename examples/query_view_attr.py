import h5py
import numpy as np

# Create simple test file
myfile = h5py.h5f.create("test_view_attr.h5")
mytype = h5py.h5t.NATIVE_INT32
myspace = h5py.h5s.create_simple((100,))
mydataset = h5py.h5d.create(myfile, "pressure", mytype, myspace)
h5py.h5d.create(myfile, "temperature", mytype, myspace)
myarray = np.zeros((100,), dtype='int32')
mydataset.write(h5py.h5s.ALL, myspace, myarray, mytype)

# Attach attribute to dataset
myattr_type = h5py.h5t.NATIVE_INT32
myattr_space = h5py.h5s.create_simple((1,))
myattr = h5py.h5a.create(mydataset, "SensorID", myattr_type, myattr_space)
myattr_array = np.array([1])
myattr.write(myattr_array, myattr_type)
myfile.close()

# Create query attr_name = "SensorID"
myquery_type = h5py.h5q.TYPE_ATTR_NAME
myquery_matchop = h5py.h5q.MATCH_EQUAL
myquery = h5py.h5q.create(myquery_type, myquery_matchop, "SensorID")

# Open file and apply query
myfile = h5py.h5f.open("test.h5", h5py.h5f.ACC_RDONLY)
myview = myquery.apply(myfile)
if h5py.h5q.REF_ATTR == myview[1]:
  print "Attribute reference found"

# Get view back (HDF5 group)
mygroup_view = myview[0]

# Read attribute reference
myref_name = h5py.h5q.VIEW_REF_ATTR_NAME
myref_dataset = h5py.h5d.open(mygroup_view, myref_name)
mynpref_type = h5py.special_dtype(ref=h5py.AttributeReference)
myref_space = h5py.h5s.create_simple((1,))
myref_array = np.empty((1,), dtype=mynpref_type)
myref_type = h5py.h5t.py_create(mynpref_type)
myref_dataset.read(h5py.h5s.ALL, myref_space, myref_array, myref_type)

# Dereference attribute reference and get object/attribute name
myderef_attr = h5py.h5r.dereference(myref_array[0], myfile)
myderef_attr_name = h5py.h5r.get_name(myref_array[0], myfile)
myderef_obj_name = h5py.h5i.get_name(myderef_attr)
print "Attribute %s from object %s matches query" % (myderef_attr_name, myderef_obj_name)


