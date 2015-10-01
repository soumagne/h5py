import h5py
import numpy as np

# Create simple test file
myfile = h5py.h5f.create("test_view_region.h5")
mytype = h5py.h5t.NATIVE_INT32
myspace = h5py.h5s.create_simple((100,))
mydataset = h5py.h5d.create(myfile, "pressure", mytype, myspace)
h5py.h5d.create(myfile, "temperature", mytype, myspace)
myarray = np.zeros((100,), dtype='int32')

# Initialize array
for i in range(0, 100):
  myarray[i] = i

mydataset.write(h5py.h5s.ALL, myspace, myarray, mytype)
myfile.close()

# Create query data_elem = "50"
myquery_type = h5py.h5q.TYPE_DATA_ELEM
myquery_matchop = h5py.h5q.MATCH_EQUAL
myquery_val = np.array([50])
myquery = h5py.h5q.create(myquery_type, myquery_matchop, mytype, myquery_val)

# Open file and apply query
myfile = h5py.h5f.open("test.h5", h5py.h5f.ACC_RDONLY)
myview = myquery.apply(myfile)
if h5py.h5q.REF_REG == myview[1]:
  print "Region reference found"

# Get view back (HDF5 group)
mygroup_view = myview[0]

# Read region reference
myref_name = h5py.h5q.VIEW_REF_REG_NAME
myref_dataset = h5py.h5d.open(mygroup_view, myref_name)
mynpref_type = h5py.special_dtype(ref=h5py.RegionReference)
myref_space = h5py.h5s.create_simple((1,))
myref_array = np.empty((1,), dtype=mynpref_type)
myref_type = h5py.h5t.py_create(mynpref_type)
myref_dataset.read(h5py.h5s.ALL, myref_space, myref_array, myref_type)

# Dereference region reference and get object name
myderef_dataset = h5py.h5r.dereference(myref_array[0], myfile)
myderef_dataset_name = h5py.h5i.get_name(myderef_dataset)
print "Dataset %s matches query" % myderef_dataset_name

# Get dataspace selection from region reference
myderef_space = h5py.h5r.get_region(myref_array[0], myfile)

# Read result value back from referenced region/dataset
myderef_sel_npoints = myderef_space.get_select_npoints()
myderef_read_sel = h5py.h5s.create_simple((myderef_sel_npoints,))
myresult = np.zeros((myderef_sel_npoints,), dtype='int32')
myderef_dataset.read(myderef_read_sel, myderef_space, myresult, mytype)
print "Element %d matches query" % myresult[0]


