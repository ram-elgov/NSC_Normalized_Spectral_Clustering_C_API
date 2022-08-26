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



static double* convert_data_points_python_to_c(PyObject *data, int n, int d) {
  int i, j;
  double *data_points;
  data_points = malloc(n * d * sizeof(double));
  for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
      data_points[i * n + j] = PyFloat_AsDouble(PyList_GetItem(data, i * n + j));
    }
  }
  return data_points;
}

static PyObject *convert_data_points_c_to_python(double matrix[], int n, int d) {
  int i, j;
  PyObject *tmp_list, *py_data_points, *num;
  py_data_points = PyList_New(n);
  for (i = 0; i < n; i++) {
    tmp_list = PyList_New(d);
    for (j = 0; j < d; j++) {
      num = PyFloat_FromDouble(matrix[i * n + j]);
      PyList_SET_ITEM(tmp_list, j, num);
    }
    PyList_SET_ITEM(py_data_points, i, tmp_list);
  }
  return py_data_points;
}

static PyObject *fit(PyObject *self, PyObject *args) {
  /* lnorm()
   * x = calculateJacobi(lnorm)
   * final_centroids kmeans++(x)
   * print(kmeans++ initial centroids)
   * print(final_centroids)
   *
   * */
  return NULL;
}

static PyObject *compute_wam(PyObject *self, PyObject *args) {
  Nsc *nsc = malloc(sizeof(Nsc));
  int length, d;
  PyObject *pointsFrompython, *WmatrixforPy, *emptylist;
  double *convertedtoCpoints;
  if (!PyArg_ParseTuple(args, "Oii", &pointsFrompython, &length, &d)) {
    emptylist = PyList_New(0);
    return emptylist;
  }
  if (!PyList_Check(pointsFrompython)) {
    emptylist = PyList_New(0);
    return emptylist;
  }
  nsc->d = d;
  nsc->n = length;
  convertedtoCpoints = convert_data_points_python_to_c(pointsFrompython, length, d);
  nsc->matrix = convertedtoCpoints;
  CalculateWeightedAdjacencyMatrix(nsc);
  WmatrixforPy = convert_data_points_c_to_python(nsc->wam, length, length);
  free(convertedtoCpoints);
  free(nsc);
  return WmatrixforPy;
}



static PyObject *compute_ddg(PyObject *self, PyObject *args) {
    int length;
    Nsc *nsc = malloc(sizeof(Nsc));
    PyObject *Wfrompython, *DforPython, *emptylist;
    double *WfromC;

    if (!PyArg_ParseTuple(args, "Oi", &Wfrompython, &length)) {
        emptylist = PyList_New(0);
        return emptylist;
    }
    if (!PyList_Check(Wfrompython)) {
        emptylist = PyList_New(0);
        return emptylist;
    }
    nsc->n = length;
    WfromC = convert_data_points_python_to_c(Wfrompython, length, length);
    nsc->matrix = WfromC;
    CalculateWeightedAdjacencyMatrix(nsc);
    CalculateDiagonalDegreeMatrix(nsc);
    DforPython = convert_data_points_c_to_python(nsc->ddg, length, length);
    free(WfromC);
    return DforPython;
}

static PyObject *compute_lnorm(PyObject *self, PyObject *args) {
  Nsc *nsc = malloc(sizeof(Nsc));
  int length, d;
  PyObject *pointsFrompython, *WmatrixforPy, *emptylist;
  double *convertedtoCpoints;
  if (!PyArg_ParseTuple(args, "Oii", &pointsFrompython, &length, &d)) {
    emptylist = PyList_New(0);
    return emptylist;
  }
  if (!PyList_Check(pointsFrompython)) {
    emptylist = PyList_New(0);
    return emptylist;
  }
  nsc->d = d;
  nsc->n = length;
  convertedtoCpoints = convert_data_points_python_to_c(pointsFrompython, length, d);
  nsc->matrix = convertedtoCpoints;
  CalculateWeightedAdjacencyMatrix(nsc);
  return NULL;
}

static PyObject *compute_jacobi(PyObject *self, PyObject *args) {
  int length, i, d, j;
  PyObject *pointsfrompy, *jacobiforpy, *emptylist;
  double *pointsforc, *values, *jacobiforc, *vectorsforc;
  if (!PyArg_ParseTuple(args, "Oii", &pointsfrompy, &length, &d)) {
    emptylist = PyList_New(0);
    return emptylist;
  }
  if (!PyList_Check(pointsfrompy)) {
    emptylist = PyList_New(0);
    return emptylist;
  }
  pointsforc = convert_data_points_python_to_c(pointsfrompy, length, d);
  values = calloc(length, sizeof(double));
  vectorsforc = CalculateJacobi(pointsforc, values, length);
  jacobiforc = calloc((length + 1) * (length + 1), sizeof(double));
  for (i = 0; i < length; i++) {
    jacobiforc[i] = values[i];
  }
  for (i = 0; i < length; i++) {
    for (j = 0; j < length; j++) {
      jacobiforc[(i + 1) * length + j] = vectorsforc[i * length + j];
    }
  }
  jacobiforpy = convert_data_points_c_to_python(jacobiforc, length + 1, length);
  free(values);
  free(pointsforc);
  free(jacobiforc);
  free(vectorsforc);
  return jacobiforpy;
}


static PyMethodDef myMethods[] = {
//    {"fit",      (PyCFunction) fit,      METH_VARARGS, PyDoc_STR("fit method for the spk algorithm")},
    {"compute_wam",    (PyCFunction) compute_wam,    METH_VARARGS, PyDoc_STR("fit method")},
//    {"compute_ddg",    (PyCFunction) compute_ddg,    METH_VARARGS, PyDoc_STR("fit method")},
//    {"compute_lnorm",  (PyCFunction) compute_lnorm,  METH_VARARGS, PyDoc_STR("fit method")},
//    {"compute_jacobi", (PyCFunction) compute_jacobi, METH_VARARGS, PyDoc_STR("fit method")},
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
