#include <functional>
#include <memory>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>

#include <immintrin.h>


#include "rectangular_lsap.h"

#ifdef __GNUC__
#define always_inline __attribute__((always_inline)) inline
#define restrict __restrict__
#elif _WIN32
#define always_inline __forceinline
#define restrict __restrict
#else
#define always_inline inline
#define restrict
#endif

#include "cpuid.h"
//#include "lap.h"

static SIMDFlags simd_flags = SIMDFlags();

static char module_docstring[] =
    "This module wraps linear sum assignment algorithm.";
static char lsa_docstring[] =
    "Solves the linear sum assignment problem.";

static PyObject *py_lsa(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"linear_sum_assignment", reinterpret_cast<PyCFunction>(py_lsa),
   METH_VARARGS | METH_KEYWORDS, lsa_docstring},
  {NULL, NULL, 0, NULL}
};

extern "C" {
PyMODINIT_FUNC PyInit_lsa(void) {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "lsa",             /* m_name */
      module_docstring,    /* m_doc */
      -1,                  /* m_size */
      module_functions,    /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PyModule_Create() failed");
    return NULL;
  }
  // numpy
  import_array();
  return m;
}
}

template <typename O>
using pyobj_parent = std::unique_ptr<O, std::function<void(O*)>>;

template <typename O>
class _pyobj : public pyobj_parent<O> {
 public:
  _pyobj() : pyobj_parent<O>(
      nullptr, [](O *p){ if (p) Py_DECREF(p); }) {}
  explicit _pyobj(PyObject *ptr) : pyobj_parent<O>(
      reinterpret_cast<O *>(ptr), [](O *p){ if(p) Py_DECREF(p); }) {}
  void reset(PyObject *p) noexcept {
    pyobj_parent<O>::reset(reinterpret_cast<O*>(p));
  }
};

using pyobj = _pyobj<PyObject>;
using pyarray = _pyobj<PyArrayObject>;

template <typename F>
static always_inline double call_lsa(int dim, const void *restrict cost_matrix, bool verbose,
                                     int *restrict row_ind, int *restrict col_ind,
                                     void *restrict u, void *restrict v) {
  double cost;
  Py_BEGIN_ALLOW_THREADS
  bool hasAVX2 = simd_flags.hasAVX2();
  if (verbose) {
    printf("AVX2: %s\n", hasAVX2? "enabled" : "disabled");
  }
  auto cost_matrix_typed = reinterpret_cast<const F*>(cost_matrix);
  auto u_typed = reinterpret_cast<F*>(u);
  auto v_typed = reinterpret_cast<F*>(v);

//  if (hasAVX2) {
//    cost = lap<true>(dim, cost_matrix_typed, verbose, row_ind, col_ind, u_typed, v_typed);
//  } else {
//    cost = lap<false>(dim, cost_matrix_typed, verbose, row_ind, col_ind, u_typed, v_typed);
//  }

  intptr_t nr = dim, nc = dim;
  double *input_cost = new double[dim*dim];
  for(int i=0;i<dim*dim;i++)input_cost[i] = cost_matrix_typed[i];
  bool maximize = false;
  int64_t* a = new int64_t[dim];
  int64_t* b = new int64_t[dim];
  solve_rectangular_linear_sum_assignment(nr, nc, input_cost, maximize, a, b);
  for(int i=0;i<dim;i++) {
    row_ind[a[i]] = b[i];
    col_ind[b[i]] = a[i];
  }
  delete[] input_cost;
  delete[] a;
  delete[] b;
  cost = 0;


  Py_END_ALLOW_THREADS
  return cost;
}

static PyObject *py_lsa(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *cost_matrix_obj;
  int verbose = 0;
  int force_doubles = 0;
  static const char *kwlist[] = {
      "cost_matrix", "verbose", "force_doubles", NULL};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|pb", const_cast<char**>(kwlist),
      &cost_matrix_obj, &verbose, &force_doubles)) {
    return NULL;
  }
  pyarray cost_matrix_array;
  bool float32 = true;
  cost_matrix_array.reset(PyArray_FROM_OTF(
      cost_matrix_obj, NPY_FLOAT32,
      NPY_ARRAY_IN_ARRAY | (force_doubles? 0 : NPY_ARRAY_FORCECAST)));
  if (!cost_matrix_array) {
    PyErr_Clear();
    float32 = false;
    cost_matrix_array.reset(PyArray_FROM_OTF(
        cost_matrix_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
    if (!cost_matrix_array) {
      PyErr_SetString(PyExc_ValueError, "\"cost_matrix\" must be a numpy array "
                                        "of float32 or float64 dtype");
      return NULL;
    }
  }
  auto ndims = PyArray_NDIM(cost_matrix_array.get());
  if (ndims != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\" must be a square 2D numpy array");
    return NULL;
  }
  auto dims = PyArray_DIMS(cost_matrix_array.get());
  if (dims[0] != dims[1]) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\" must be a square 2D numpy array");
    return NULL;
  }
  int dim = dims[0];
  if (dim <= 0) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\"'s shape is invalid or too large");
    return NULL;
  }
  auto cost_matrix = PyArray_DATA(cost_matrix_array.get());
  npy_intp ret_dims[] = {dim, 0};
  pyarray row_ind_array(PyArray_SimpleNew(1, ret_dims, NPY_INT));
  pyarray col_ind_array(PyArray_SimpleNew(1, ret_dims, NPY_INT));
  auto row_ind = reinterpret_cast<int*>(PyArray_DATA(row_ind_array.get()));
  auto col_ind = reinterpret_cast<int*>(PyArray_DATA(col_ind_array.get()));
  pyarray u_array(PyArray_SimpleNew(
      1, ret_dims, float32? NPY_FLOAT32 : NPY_FLOAT64));
  pyarray v_array(PyArray_SimpleNew(
      1, ret_dims, float32? NPY_FLOAT32 : NPY_FLOAT64));
  double cost;
  auto u = PyArray_DATA(u_array.get());
  auto v = PyArray_DATA(v_array.get());
  if (float32) {
    cost = call_lsa<float>(dim, cost_matrix, verbose, row_ind, col_ind, u, v);
  } else {
    cost = call_lsa<double>(dim, cost_matrix, verbose, row_ind, col_ind, u, v);
  }
  return Py_BuildValue("(OO)",
                       row_ind_array.get(), col_ind_array.get());
}
