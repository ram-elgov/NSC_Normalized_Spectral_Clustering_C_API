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
/* C API code kmeans */

static void free_2d_arr(double** m, int rows) {
/*
function to free memory allocated for 2d array.
*/
  int i;
  for (i = 0; i < rows; ++i) {
    free(m[i]);
  }
  free(m);
}
static int is_converged(double *centroids, double *old_centroids, int K, int d, double epsilon) {
/*
checks if the convergence criteria has been reached.
*/
  int i, j;
  double norm;
  /*
   * checks if all centroid's norm didn't change more than the value of epsilon. if yes, then it's converged.
   */
  for (i = 0; i < K; i++) {
    norm = 0;
    for (j = 0; j < d; j++) {
      norm += pow(centroids[i * d + j] - old_centroids[i * d + j], 2);
    }
    norm = pow(norm, 0.5);
    if (norm >= epsilon) {
      return 1;
    }
  }
  return 0;
}
static int index_of_closest_cluster(double *x, double *centroids, int K, int d) {
/*
calculating the index of the closest cluster to the given data point.
*/
  double min = 0;
  double sum;
  int i, j, index = 0;
  /* minimum initialisation. (the fis  */
  for (i = 0; i < d; i++) {
    min += pow(x[i] - centroids[i], 2);
  }
  /* checks for the rest of the centroids. */
  for (j = 0; j < K; j++) {
    sum = 0;
    for (i = 0; i < d; i++) {
      sum += pow(x[i] - centroids[j * d + i], 2);
    }
    if (sum < min) {
      min = sum;
      index = j;
    }
  }
  return index;
}
static int run(double** data_points, double** centroids_pp, int n, int d, int k, int max_iter, double epsilon) {
/*
the main clustering algorithm using kmeans.
same implementation from HW1 except using kmeans++ and data parsing implemented in python.
*/
  int iteration_num, valid, i, j, t;
  double *vectors;
  double *centroids;
  double *old_centroids;
  double *clusters;
  int *sizeof_clusters;
  centroids = calloc(k * d, sizeof(double));
  vectors = calloc(d * n, sizeof(double));
  old_centroids = calloc(k * d, sizeof(double));
  clusters = calloc(k * d, sizeof(double));
  sizeof_clusters = calloc(k, sizeof(int));

  // kmeans++ centroids initialization ------------------------------------------------------
  t = 0;
  i = 0;
  j = 0;
  while (t < d * k) {
    if (j == d) {
      j = 0;
      ++i;
    }
    if(i == k) break;
    centroids[t] = centroids_pp[i][j];
    ++j;
    ++t;
  }
  // -----------------------------------------------------------------------------------
  iteration_num = 0;
  valid = 1;
  while (iteration_num < max_iter && valid == 1) {
    for (i = 0; i < k * d; i++) {
      old_centroids[i] = centroids[i];
    }
    for (i = 0; i < n; i++) {
      int index = index_of_closest_cluster(data_points[i], centroids, k, d);
      for (j = 0; j < d; j++) {
        clusters[index * d + j] += data_points[i][j];
      }
      sizeof_clusters[index]++;
    }
    for (j = 0; j < k; j++) {
      for (i = 0; i < d; i++) {
        centroids[d * j + i] = clusters[d * j + i] / sizeof_clusters[j];
      }
    }
    for (j = 0; j < k * d; j++) {
      clusters[j] = 0;
    }
    for (j = 0; j < k; j++) {
      sizeof_clusters[j] = 0;
    }
    valid = is_converged(centroids, old_centroids, k, d, epsilon);
    iteration_num++;
  }
  t = 0;
  i = 0;
  j = 0;
  while (t < d * k) {
    if (j == d) {
      j = 0;
      ++i;
    }
    if(i == k) break;
    centroids_pp[i][j] = centroids[t];
    ++j;
    ++t;
  }
  free(clusters);
  free(sizeof_clusters);
  free(centroids);
  free(old_centroids);
  free(vectors);
  return 0;
}
/*
C API code
*/
static double** get_from_python(int num_of_elements, int dim, PyObject *python_list){
/*
parse python list input into 2d array.
*/
  int i, j;
  double **matrix;
  PyObject *temp_list, *element;
  matrix = calloc(num_of_elements, sizeof(double*));
  for (i = 0; i < num_of_elements; i++){
    matrix[i] = calloc(dim, sizeof(double));
    temp_list = PyList_GetItem(python_list, i);
    for (j = 0; j < dim; j++){
      element = PyList_GetItem(temp_list, j);
      matrix[i][j] = PyFloat_AsDouble(element);
    }
  }
  return matrix;
}
static PyObject* send_to_python(double** centroids, int K, int dim){
/*
send the final centroids to python as a list object.
*/
  int i, j;
  PyObject* outer_list;
  PyObject* inner_list;
  PyObject* element;
  outer_list = PyList_New(K);
  for (i = 0; i < K; i++){
    inner_list = PyList_New(dim);
    for (j = 0; j < dim; j++){
      element = PyFloat_FromDouble(centroids[i][j]);
      PyList_SET_ITEM(inner_list, j, element);
    }
    PyList_SET_ITEM(outer_list, i, inner_list);
  }
  return outer_list;
}
static PyObject* fit_kmeans(PyObject *self, PyObject *args) {
/*
the algorithm's fit() function. calls run() and return the output back to python.
*/
  PyObject *output, *data_points_list, *centroid_list;
  int N, K, max_iter, dim;
  double **centroids, **data_points;
  double epsilon;
  if (!PyArg_ParseTuple(args, "iiiidOO", &N, &K, &max_iter, &dim, &epsilon,
                        &centroid_list, &data_points_list)){
    return NULL;
  }
  data_points = get_from_python(N, dim, data_points_list);
  centroids = get_from_python(K, dim, centroid_list);
  if (run(data_points, centroids, N, dim, K, max_iter, epsilon))
  {
    free_2d_arr(data_points, N);
    free_2d_arr(centroids, K);
    return NULL;
  }
  else {
    output = send_to_python(centroids, K, dim);
    free_2d_arr(data_points, N);
    free_2d_arr(centroids, K);
    return output;
  }
}
/*
C API code spectral clustering
*/

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

static PyObject *fit(PyObject *self, PyObject *args) {
  /* Declarations */
  Nsc nsc;
  PyObject *empty_list, *data_points_from_python,
      *result_for_python = PyTuple_New(2);
  double *data_points_converted_to_c, *t, *u;
  int n, d, k;
  /* Parsing arguments */
  if (!PyArg_ParseTuple(args,
                        "Oiii",
                        &data_points_from_python,
                        &n,
                        &d,
                        &k)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  if (!PyList_Check(data_points_from_python)) {
    empty_list = PyList_New(0);
    return empty_list;
  }
  /* Memory allocation */
  AllocateMatrix(&data_points_converted_to_c, n, d);
  /* Data points conversion */
  convert_object_python_to_c(data_points_from_python,
                             data_points_converted_to_c, n, d);
  /* Initialize the Nsc object */
  ConstructNsc(&nsc, data_points_converted_to_c, n, d, FIT);
  /* Preform the spectral clustering steps */
  CalculateNormalizedGraphLaplacian(&nsc);
  CalculateJacobi(&nsc);
  /* Calculates k and sorts eigen_vectors and eigen_values */
  k = FindK(&nsc, k);
  AllocateMatrix(&u, n, k);
  AllocateMatrix(&t, n, k);
  CalculateUMatrix(&nsc, u, k);
  CalculateTMatrix(u, t, n, k);
  /* Convert output to a python object */
  PyTuple_SetItem(result_for_python, 0,
                  convert_object_c_to_python(t, n, k));
  PyTuple_SetItem(result_for_python, 1, PyLong_FromLong(k));
  /* Memory de-allocation */
  FreeMatrix(&data_points_converted_to_c);
  FreeMatrix(&u);
  FreeMatrix(&t);
  DestructNsc(&nsc);
  /* Return the computed t matrix as a python object */
  return result_for_python;
}

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
  convert_object_python_to_c(data_points_from_python,
                             data_points_converted_to_c,
                             n,
                             d);
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
  convert_object_python_to_c(data_points_from_python,
                             data_points_converted_to_c,
                             n,
                             d);
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

static PyObject *compute_jacobi(PyObject *self, PyObject *args) {
  /* Declarations */
  Nsc nsc;
  int n, d;
  PyObject * data_points_from_python, *result_for_python, *empty_list;
  double *data_points_converted_to_c, *jacobi_result;
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
  AllocateMatrix(&jacobi_result, n + 1, n);
  /* Conversion python to c and initialization of data structure */
  convert_object_python_to_c(data_points_from_python,
                             data_points_converted_to_c, n, d);
  ConstructNsc(&nsc, data_points_converted_to_c, n, d, JACOBI);
  /* Algorithm calculation */
  CalculateJacobi(&nsc);
  CopyMatrix(jacobi_result, nsc.eigen_values, 1, d);
  CopyMatrix(&jacobi_result[d], nsc.eigen_vectors, n, n);
  /* Conversion c to python */
  result_for_python = convert_object_c_to_python(jacobi_result, n + 1, n + 1);
  /* Memory de-allocation */
  FreeMatrix(&data_points_converted_to_c);
  FreeMatrix(&jacobi_result);
  DestructNsc(&nsc);
  /* Return */
  return result_for_python;
}

static PyMethodDef myMethods[] = {
    {"fit", (PyCFunction) fit, METH_VARARGS,
     PyDoc_STR("fit method for the spk algorithm")},
    {"fit_kmeans", (PyCFunction) fit_kmeans,METH_VARARGS, PyDoc_STR("runs "
                                                                    "the "
                                                                    "kmeans "
                                                                    "algorithem"
                                                                    "") },
    {"compute_wam", (PyCFunction) compute_wam, METH_VARARGS,
     PyDoc_STR("wam method")},
    {"compute_ddg", (PyCFunction) compute_ddg, METH_VARARGS,
     PyDoc_STR("fit method")},
    {"compute_lnorm", (PyCFunction) compute_lnorm, METH_VARARGS,
     PyDoc_STR("fit method")},
    {"compute_jacobi", (PyCFunction) compute_jacobi, METH_VARARGS,
     PyDoc_STR("fit method")},
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
