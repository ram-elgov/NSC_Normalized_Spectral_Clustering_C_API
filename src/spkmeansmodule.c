#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"
/*****************************************************************************

@author: mohammad daghash
@id: 314811290
@author: ram elgov
@id: 206867517

Implementation of the Normalized Spectral Clustering algorithm.

******************************************************************************/



static void convert_object_python_to_c(PyObject *data_points_from_py,
                                       double data_points_converted_to_c[],
                                       int n,
                                       int d) {
  int i, j;
  for (i = 0; i < n; ++i)
    for (j = 0; j < d; ++j) {
      data_points_converted_to_c[i * d + j] = PyFloat_AsDouble(
          PyList_GetItem(data_points_from_py, i * d + j));
    }
}

static PyObject *convert_object_c_to_python(double *matrix, int n, int d) {
  int i, j;
  PyObject * pyMatrix;
  pyMatrix = PyList_New(n * d);
  for (i = 0; i < n; ++i) {
    for (j = 0; j < d; ++j) {
      PyList_SET_ITEM(pyMatrix, i * d + j,
                      PyFloat_FromDouble(matrix[i * d + j]));
    }
  }
  return pyMatrix;
}

/*static PyObject *fit(PyObject *self, PyObject *args) {
  Nsc *nsc;
  double epsilon;
  double *t;
  PyObject *empty_list, *data_points_from_python;
  int n, d, k;
  if (!PyArg_ParseTuple(args, "Oiiid", &data_points_from_python, &n, &d, &k, &epsilon)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  if (!PyList_Check(data_points_from_python)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  NSC(nsc, convert_data_points_python_to_c(data_points_from_python,n,d), n, d, epsilon);
  CalculateJacobi(nsc);
  k = FindK(nsc, k);
  t = TMatrix(  UMatrix(nsc, k), n, k);
  return convert_data_points_c_to_python(t, nsc->n, k);
  return PyLong_FromLong(1998);
}*/
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedParameter"
static PyObject *compute_wam(PyObject *self, PyObject *args) {
#pragma clang diagnostic pop
  /* Declarations */
  Nsc nsc;
  int n, d;
  double *data_points_converted_to_c;
  /* Parsing arguments */
  PyObject * data_points_from_python, *result_for_python, *empty_list;
  if (!PyArg_ParseTuple(args, "Oii", &data_points_from_python, &n, &d)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  if (!PyList_Check(data_points_from_python)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  /* Memory allocation */
  AllocateMatrix(&data_points_converted_to_c, n, d);
  /* Conversion python to c and initialization of data structure */
  convert_object_python_to_c(data_points_from_python, data_points_converted_to_c, n, d);
  ConstructNsc(&nsc, data_points_converted_to_c, n, d, WAM);
  /* Algorithm calculation */
  CalculateWeightedAdjacencyMatrix(&nsc);
  /* Conversion c to python */
  result_for_python = convert_object_c_to_python(nsc.wam, n, n);
  /* Memory de-allocation */
  FreeMatrix(&data_points_converted_to_c);
  DestructNsc(&nsc);
  /* Return */
  return result_for_python;
}

static PyObject *compute_ddg(PyObject *self, PyObject *args) {
  /* Declarations */
  int n, d;
  Nsc nsc;
  double *data_points_converted_to_c;
  PyObject * data_points_from_python, *result_for_python, *empty_list;
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "Oii", &data_points_from_python, &n, &d)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  if (!PyList_Check(data_points_from_python)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  /* Memory allocation */
  AllocateMatrix(&data_points_converted_to_c, n, d);
  /* Conversion python to c and initialization of data structure */
  convert_object_python_to_c(
      data_points_from_python, data_points_converted_to_c, n, d);
  ConstructNsc(&nsc, data_points_converted_to_c, n, d, DDG);
  /* Algorithm calculation */
  CalculateDiagonalDegreeMatrix(&nsc);
  /* Conversion c to python */
  result_for_python = convert_object_c_to_python(nsc.ddg, n, n);
  /* Memory de-allocation */
  FreeMatrix(&data_points_converted_to_c);
  DestructNsc(&nsc);
  /* Return */
  return result_for_python;
}

static PyObject *compute_lnorm(PyObject *self, PyObject *args) {
  /* Declarations */
  Nsc nsc;
  int n, d;
  PyObject * data_points_from_python, *result_for_python, *empty_list;
  double *data_points_converted_to_c;
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args, "Oii", &data_points_from_python, &n, &d)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  if (!PyList_Check(data_points_from_python)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  /* Memory allocation */
  AllocateMatrix(&data_points_converted_to_c, n, d);
  /* Conversion python to c and initialization of data structure */
  convert_object_python_to_c(data_points_from_python, data_points_converted_to_c, n, d);
  ConstructNsc(&nsc, data_points_converted_to_c, n, d, LNORM);
  /* Algorithm calculation */
  CalculateNormalizedGraphLaplacian(&nsc);
  /* Conversion c to python */
  result_for_python = convert_object_c_to_python(nsc.l_norm, n, n);
  /* Memory de-allocation */
  FreeMatrix(&data_points_converted_to_c);
  DestructNsc(&nsc);
  /* Return */
  return result_for_python;
}

//static PyObject *compute_jacobi(PyObject *self, PyObject *args) {
//  int n, i, d, j;
//  PyObject *points_from_py, *jacobi_for_py, *empty_list;
//  double *points_for_c, *jacobi_for_c;
//  Nsc *nsc = malloc(sizeof(Nsc));
//  assert(nsc != NULL);
//  if (!PyArg_ParseTuple(args, "Oii", &points_from_py, &n, &d)) {
//    empty_list = PyList_New(0);
//    return empty_list;
//  }
//  if (!PyList_Check(points_from_py)) {
//    empty_list = PyList_New(0);
//    return empty_list;
//  }
//  points_for_c = convert_data_points_python_to_c(points_from_py, n, d);
//  nsc->d = d;
//  nsc->n = n;
//  nsc->matrix = points_for_c;
//  nsc->eigen_values = calloc(n, sizeof(double));
//  assert(nsc->eigen_values != NULL);
//  CalculateJacobi(nsc);
//  jacobi_for_c = calloc((n + 1) * (n + 1), sizeof(double));
//  assert(jacobi_for_c != NULL);
//  for (i = 0; i < n; i++) {
//    jacobi_for_c[i] = nsc->eigen_values[i];
//  }
//  for (i = 0; i < n; i++) {
//    for (j = 0; j < n; j++) {
//      jacobi_for_c[(i + 1) * n + j] = nsc->eigen_vectors[i * n + j];
//    }
//  }
//  jacobi_for_py = convert_data_points_c_to_python(jacobi_for_c, n + 1, n);
//  free(nsc->eigen_values);
//  free(nsc->eigen_vectors);
//  free(points_for_c);
//  free(jacobi_for_c);
//  return jacobi_for_py;
//}


static PyMethodDef myMethods[] = {
//    {"fit",      (PyCFunction) fit,      METH_VARARGS, PyDoc_STR("fit method for the spk algorithm")},
    {"compute_wam", (PyCFunction) compute_wam, METH_VARARGS, PyDoc_STR("wam method")},
    {"compute_ddg", (PyCFunction) compute_ddg, METH_VARARGS, PyDoc_STR("fit method")},
    {"compute_lnorm", (PyCFunction) compute_lnorm, METH_VARARGS, PyDoc_STR("fit method")},
    /*{"compute_jacobi", (PyCFunction) compute_jacobi, METH_VARARGS, PyDoc_STR("fit method")},*/
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef finalmodule = {
    PyModuleDef_HEAD_INIT,
    "finalmodule",
    NULL,
    -1,
    myMethods
};

PyMODINIT_FUNC
PyInit_finalmodule(void) {
  return PyModule_Create(&finalmodule);
}
