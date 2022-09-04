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



static double **convert_data_points_python_to_c(PyObject *data_points_from_py, int n, int d) {
  int i, j;
  double **points;
  PyObject *helppoint, *item;
  double *point;
  points = calloc(n, sizeof(double *));
  for (i = 0; i < n; i++) {
    point = calloc(d, sizeof(double));
    helppoint = PyList_GetItem(data_points_from_py, i);
    for (j = 0; j < d; j++) {
      item = PyList_GetItem(helppoint, j);
      point[j] = PyFloat_AsDouble(item);
    }
    points[i] = point;
  }
  return points;
}

static PyObject *convert_data_points_c_to_python(double **matrix, int n, int d) {
  int i, j;
  PyObject *helpVector, *pyMatrix, *num;
  pyMatrix = PyList_New(n);
  for (i = 0; i < n; i++) {
    helpVector = PyList_New(d);
    for (j = 0; j < d; j++) {
      num = PyFloat_FromDouble(matrix[i][j]);
      PyList_SET_ITEM(helpVector, j, num);
    }
    PyList_SET_ITEM(pyMatrix, i, helpVector);
  }
  return pyMatrix;
}

/* java like syntax for constructor */
Nsc *NSC(Nsc *nsc, double data_points[], int n, int d, double epsilon) {
  nsc = malloc(sizeof(Nsc));
  assert(nsc != NULL);
  nsc->epsilon = epsilon;
  nsc->n = n;
  nsc->d = d;
  nsc->matrix = malloc((nsc->n * nsc->d) * sizeof(double));
  assert(nsc->matrix != NULL);
  CopyMatrix(nsc->matrix, data_points,nsc->n, nsc->d);
  nsc->wam = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->wam != NULL);
  nsc->eigen_values = malloc(nsc->n * sizeof(double));
  assert(nsc->eigen_values != NULL);
  nsc->eigen_vectors = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->eigen_vectors != NULL);
  nsc->ddg = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->ddg != NULL);
  nsc->inversed_sqrt_ddg = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->inversed_sqrt_ddg != NULL);
  return nsc;
}

static PyObject *fit(PyObject *self, PyObject *args) {
  /*Nsc *nsc;
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
  return convert_data_points_c_to_python(t, nsc->n, k);*/
  return PyLong_FromLong(1998);
}
//
static PyObject *compute_wam(PyObject *self, PyObject *args) {
  Nsc *nsc;
  int n, d;
  double epsilon;
  double **data_points_converted_to_c;
  double *data_points_converted_to_c_1d;
  PyObject *pointsFrompython, *WmatrixforPy, *empty_list;
  if (!PyArg_ParseTuple(args, "Oiid", &pointsFrompython, &n, &d, &epsilon)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  if (!PyList_Check(pointsFrompython)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  data_points_converted_to_c = convert_data_points_python_to_c(pointsFrompython, n, d);
  data_points_converted_to_c_1d = Convert2dto1d(data_points_converted_to_c, n, d);
  nsc = NSC(nsc, data_points_converted_to_c_1d , n, d, epsilon);
  CalculateWeightedAdjacencyMatrix(nsc);
  PrintMatrix(nsc->wam, n,n);
//  WmatrixforPy = convert_data_points_c_to_python(Convert1dto2d(nsc->wam, n,n), n, n);
  printf("hi\n");
  return PyList_New(0);
}
//
//static PyObject *compute_ddg(PyObject *self, PyObject *args) {
//    int n;
//    Nsc *nsc;
//    PyObject *Wfrompython, *DforPython, *empty_list;
//    double *WfromC;
//    if (!PyArg_ParseTuple(args, "Oi", &Wfrompython, &n)) {
//        empty_list = PyList_New(0);
//        return empty_list;
//    }
//    if (!PyList_Check(Wfrompython)) {
//        empty_list = PyList_New(0);
//        return empty_list;
//    }
//    NSC(nsc, convert_data_points_python_to_c(Wfrompython, n, n), n, n, 0);
//    CalculateWeightedAdjacencyMatrix(nsc);
//    CalculateDiagonalDegreeMatrix(nsc);
//    DforPython = convert_data_points_c_to_python(nsc->ddg, n, n);
//    free(WfromC);
//    return DforPython;
//
//}
//
//static PyObject *compute_lnorm(PyObject *self, PyObject *args) {
//  Nsc *nsc = malloc(sizeof(Nsc));
//  int length, d;
//  PyObject *pointsFrompython, *WmatrixforPy, *empty_list;
//  double *convertedtoCpoints;
//  if (!PyArg_ParseTuple(args, "Oii", &pointsFrompython, &length, &d)) {
//    empty_list = PyList_New(0);
//    return empty_list;
//  }
//  if (!PyList_Check(pointsFrompython)) {
//    empty_list = PyList_New(0);
//    return empty_list;
//  }
//  nsc->d = d;
//  nsc->n = length;
//  convertedtoCpoints = convert_data_points_python_to_c(pointsFrompython, length, d);
//  nsc->matrix = convertedtoCpoints;
//  CalculateWeightedAdjacencyMatrix(nsc);
//  return NULL;
//}
//
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
    {"fit",      (PyCFunction) fit,      METH_VARARGS, PyDoc_STR("fit method for the spk algorithm")},
    {"compute_wam",    (PyCFunction) compute_wam,    METH_VARARGS, PyDoc_STR("fit method")},
    /*{"compute_ddg",    (PyCFunction) compute_ddg,    METH_VARARGS, PyDoc_STR("fit method")},
    {"compute_lnorm",  (PyCFunction) compute_lnorm,  METH_VARARGS, PyDoc_STR("fit method")},
    {"compute_jacobi", (PyCFunction) compute_jacobi, METH_VARARGS, PyDoc_STR("fit method")},*/
    {NULL, NULL,                           0, NULL}
};

static struct PyModuleDef spkmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule",
    NULL,
    -1,
    myMethods
};

PyMODINIT_FUNC
PyInit_spkmeansmodule(void) {
  return PyModule_Create(&spkmeansmodule);
}
