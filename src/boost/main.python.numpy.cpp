#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
// Include the headers of MyLib

namespace np = boost::python::numpy;
namespace p = boost::python;

BOOST_PYTHON_MODULE(boost_numpy)
{
    Py_Initialize();
    np::initialize();

    // Write the bindings here

p::tuple shape = p::make_tuple(3, 3);
np::dtype dtype = np::dtype::get_builtin<float>();
np::ndarray a = np::zeros(shape, dtype);
np::ndarray b = np::empty(shape,dtype);


  std::cout << "Original array:\n" << p::extract<char const *>(p::str(a)) << std::endl;

  // Reshape the array into a 1D array
  a = a.reshape(p::make_tuple(9));

 for(int i=0; i<9; i++) a[i] = i;

  // Print it again.
  std::cout << "Reshaped array:\n" << p::extract<char const *>(p::str(a)) << std::endl;

  

}

