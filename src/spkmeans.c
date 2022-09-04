#include "stdio.h"
#include "stdlib.h"
#include "spkmeans.h"
#include "assert.h"
#include "string.h"
#include "math.h"
/******************************************************************************

@author: mohammad daghash
@id: 314811290
@author: ram elgov
@id: 206867517

Implementation of the Normalized Spectral Clustering algorithm.

*******************************************************************************/

/* standalone client */
int main(int argc, char **argv) {
  /* declarations */
  double *data_points;
  int n, d;
  Goal user_goal; /* Goal is an enum. see header for more info. */
  Nsc nsc; /* Nsc is a struct representing
 * a data structure to support all the possible goals. */

  /* argument processing and validation */
  if (argc != 3) {
    InvalidInput();
    exit(1);
  }
  if (!strcmp(argv[1], "wam")) {
    user_goal = WAM;
  } else if (!strcmp(argv[1], "ddg")) {
    user_goal = DDG;
  } else if (!strcmp(argv[1], "lnorm")) {
    user_goal = LNORM;
  } else if (!strcmp(argv[1], "jacobi")) {
    user_goal = JACOBI;
  } else {
    InvalidInput();
    exit(1);
  }
  /* Parsing the input file and initialize a Nsc data structure */

  CalculateNandD(argv[2], &n, &d);
  AllocateMatrix(&data_points, n, d);
  BuildDataPointsMatrix(argv[2],data_points, n,d);
  ConstructNsc(&nsc,data_points, n, d, user_goal);
  /* run the required calculation based on the given goal */
  ChooseGoal(&nsc);
  /* Used memory de-allocation */
  FreeMatrix(&data_points);
  DestructNsc(&nsc);
  return 0;
}


void PrintMatrix(const double *matrix, int n, int d) {
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < d; ++j) {
      printf("%.4f", matrix[i * d + j]);
      if (j != d - 1)
        printf(",");
    }
    printf("\n");
  }
}
void AllocateMatrix(double **matrix, int n, int d) {
  *matrix = calloc(n * d, sizeof(double));
  assert((*matrix) != NULL);
}
void FreeMatrix(double **matrix) {
  free(*matrix);
  *matrix = NULL;
}
void PrintMatrixJacobi(Nsc *nsc) {
  int i;
  int n = nsc->n;
  for (i = 0; i < n; i++) {
    if (nsc->eigen_values[i] > -0.00001 && nsc->eigen_values[i] < 0) {
      if (i < n - 1) {
        printf("%.4f%s", fabs(nsc->eigen_values[i]), ",");
      } else {
        printf("%.4f\n", fabs(nsc->eigen_values[i]));
      }
    } else {
      if (i < n - 1) {
        printf("%.4f%s", nsc->eigen_values[i], ",");
      } else {
        printf("%.4f\n", nsc->eigen_values[i]);
      }
    }
  }
}
void InvalidInput() {
  printf("Invalid Input!");
}
void GeneralError() {
  printf("An Error Has Occurred");
}
void ChooseGoal(Nsc *nsc) {
  switch (nsc->goal) {
    case WAM:CalculateWeightedAdjacencyMatrix(nsc);
      PrintMatrix(nsc->wam, nsc->n, nsc->n);
      break;
    case DDG:CalculateDiagonalDegreeMatrix(nsc);
      PrintMatrix(nsc->ddg, nsc->n, nsc->n);
      break;
    case LNORM:CalculateNormalizedGraphLaplacian(nsc);
      PrintMatrix(nsc->l_norm, nsc->n, nsc->n);
      break;
    case JACOBI:CalculateJacobi(nsc);
//      PrintMatrix(nsc->eigen_values, 1, nsc->n);
//      PrintMatrix(nsc->eigen_vectors, nsc->n, nsc->n);
      break;
  }
}

/*
 * library functions implementation
 */

void CalculateWeightedAdjacencyMatrix(Nsc *nsc) {
  double w_ij;
  int i, j;
  /* calculates the Weighted Adjacency Matrix.
   * https://moodle.tau.ac.il/mod/forum/discuss.php?d=127889
   * - use standard euclidean norm as defined in the project specification */
  /* we do not allow self loops, so we set w_ii = 0 for all I’s */

  for (i = 0; i < nsc->n; ++i) {
    (nsc->wam)[i * nsc->n + i] = 0; /* wam[i][i] = 0 */
  }

  /* assign weights with respect to symmetry
   * running on upper triangle indices only */
  for (i = 0; i < nsc->n; ++i) {
    for (j = i + 1; j < nsc->n; ++j) {
      /* helper function to calculate the weight */
      w_ij = CalculateWeight(i, j, nsc);
      (nsc->wam)[i * nsc->n + j] = w_ij; /* wam[i][j] = w_ij */
      (nsc->wam)[j * nsc->n + i] = w_ij; /* wam[i][j] = w_ij */
    }
  }
}
/* Calculate and output the Diagonal Degree Matrix as described in 1.1.2. */
void CalculateDiagonalDegreeMatrix(Nsc *nsc) {
  double val = 0;
  int i, j, n = nsc->n;
  CalculateWeightedAdjacencyMatrix(nsc);
  for (i = 0; i < n; i++) {
    val = 0;
    for (j = 0; j < n; j++) {
      val += (nsc->wam)[i * n + j];
    }
    (nsc->ddg)[i * nsc->n + i] = val;
  }
}
void CalculateNormalizedGraphLaplacian(Nsc *nsc) {
  /* Declarations */
  double *identity;
  double *inversed_dw;
  double *inversed_dw_inversed_d;
  int n = nsc->n;
  /* Memory allocation */
  AllocateMatrix(&identity, n, n);
  AllocateMatrix(&inversed_dw, n, n);
  AllocateMatrix(&inversed_dw_inversed_d, n, n);
  /* Run the first three steps of the algorithm using
   * the existing goals implementations. */
  CalculateWeightedAdjacencyMatrix(nsc);
  CalculateDiagonalDegreeMatrix(nsc);
  InversedSqrtDiagonalDegreeMatrix(nsc);
  IdentityMatrix(identity, n);
  MultiplyTwoMatrices(nsc->inversed_sqrt_ddg, nsc->wam, inversed_dw, n);
  MultiplyTwoMatrices(inversed_dw, nsc->inversed_sqrt_ddg, inversed_dw_inversed_d, n);
  SubTwoMatrices(identity, inversed_dw_inversed_d, nsc->l_norm, n);
  /* Memory de-allocation*/
  FreeMatrix(&identity);
  FreeMatrix(&inversed_dw);
  FreeMatrix(&inversed_dw_inversed_d);
}
/**
 * Procedure:
(a) Build a rotation matrix P (as explained below).
(b) Transform the matrix A to:
A' = P^TAP
A = A'
(c) Repeat a,b until A' is diagonal matrix.
(d) The diagonal of the final A 0 is the eigenvalues of the original A.
(e) Calculate the eigenvectors of A by multiplying all the rotation matrices:
V = P 1 P 2 P 3 . . .
 * */
void CalculateJacobi(Nsc *nsc) {
  /* Declerations */
  double *p, *a_tag, *a, *v;
  /* v is the product of all rotation matrices p1p2p3... */
  int i, n = nsc->n;
  /* Memory allocation */
  AllocateMatrix(&a, n, n);
  CopyMatrix(a, nsc->l_norm, n, n);
  AllocateMatrix(&p, n, n);
  AllocateMatrix(&a_tag, n, n);
  CopyMatrix(a_tag, a, n, n);
  AllocateMatrix(&v, n, n);
  /* Preform calculations until epsilon convergence or
   * num_iteration exceeds 100 */
  RunJacobiCalculations(a,a_tag, p, v, n, nsc);
  /* Extract results */
  for (i = 0; i < nsc->n; ++i) {
    nsc->eigen_values[i] = a[i * n + i];
  }
  /* Memory de-allocation */
  FreeMatrix(&a);
  FreeMatrix(&a_tag);
  FreeMatrix(&p);
  FreeMatrix(&v);
}

/*
 * API helper functions
 */

void InversedSqrtDiagonalDegreeMatrix(Nsc *nsc) {
  double val;
  int i;
  for (i = 0; i < nsc->n; i++) {
    val = (nsc->ddg)[i * nsc->n + i];
    (nsc->inversed_sqrt_ddg)[i * nsc->n + i] = 1 / (sqrt(val));
  }
}
void CalculateRotationMatrix(const double a[], double p[], int n, Nsc *nsc) {
  int i_pivot, j_pivot;
  double c, s, theta, t, pivot;
  pivot = a[1];
  i_pivot = 0;
  j_pivot = 1;
  IdentityMatrix(p, n); /* P = I */
  FindPivot(a, n, &pivot, &i_pivot, &j_pivot);
  theta = (a[j_pivot * n + j_pivot] - a[i_pivot * n + i_pivot]) / (2 * pivot);
  t = Sign(theta) / (fabs(theta) + sqrt(theta * theta + 1));
  c = 1 / (sqrt(t * t + 1));
  s = t * c;
  p[i_pivot * n + i_pivot] = c;
  p[j_pivot * n + j_pivot] = c;
  p[i_pivot * n + j_pivot] = s;
  p[j_pivot * n + i_pivot] = (-1) * s;
  nsc->s = s;
  nsc->c = c;
  nsc->i_pivot = i_pivot;
  nsc->j_pivot = j_pivot;
}
void RunJacobiCalculations(double a[], double a_tag[], double p[], double v[], int n,  Nsc *nsc) {
  /* Declerations */
  int num_iteration = 0;
  double convergence = nsc->epsilon + 1;
  /* v is the partial product of rotation matrecies p1p2p3... */
  IdentityMatrix(nsc->eigen_vectors, n);
  while (num_iteration < 100 && convergence > nsc->epsilon) {
    CalculateRotationMatrix(a, p, n, nsc); /* p is the rotation matrix for a */
    PrintMatrix(p, n, n);
    CalculateATagEfficient(a, a_tag, nsc);
    CopyMatrix(a, a_tag, n, n);
    CopyMatrix(v, nsc->eigen_vectors, n, n);
    MultiplyTwoMatrices(v,p,nsc->eigen_vectors,n);
    convergence = (Off(a, n) - Off(a_tag, n));
    ++num_iteration;
  }
}
void FindPivot(const double a[], int n, double *pivot, int *i_pivot, int *j_pivot) {
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = i + 1; j < n; ++j) {
      if (fabs(a[i * n + j]) > fabs(*pivot)) {
        *pivot = a[i * n + j];
        *i_pivot = i;
        *j_pivot = j;
      }
    }
  }
}
int Sign(double theta) {
  return theta >= 0 ? 1 : -1;
}
double Off(double a[], int n) {
  double off = 0.0;
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      if (i != j) {
        off += pow(a[i * n + j], 2);
      }
    }
  }
  return off;
}
void CopyMatrix(double a[], const double b[], int n, int d) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
      a[i * d + j] = b[i * d + j];
    }
  }
}
void ConstructNsc(Nsc *nsc, double *data_points, int n, int d, Goal goal) {
  nsc->goal = goal;
  nsc->n = n;
  nsc->d = d;
  AllocateMatrix(&(nsc->matrix), nsc->n, nsc->d);
  CopyMatrix(nsc->matrix, data_points, n, d);
  AllocateMatrix(&(nsc->wam), nsc->n, nsc->n);
  if (nsc->goal == WAM)
    return;
  AllocateMatrix(&(nsc->ddg), nsc->n, nsc->n);
  if (nsc->goal == DDG)
    return;
  AllocateMatrix(&(nsc->inversed_sqrt_ddg), nsc->n, nsc->n);
  AllocateMatrix(&(nsc->l_norm), nsc->n, nsc->n);
  if (nsc->goal == LNORM)
    return;
  AllocateMatrix(&(nsc->eigen_values), 1, n);
  AllocateMatrix(&(nsc->eigen_vectors), n, n);

}
void DestructNsc(Nsc *nsc) {
  FreeMatrix(&(nsc->matrix));
  FreeMatrix(&(nsc->wam));
  if (nsc->goal == WAM)
    return;
  FreeMatrix(&(nsc->ddg));
  if (nsc->goal == DDG)
    return;
  FreeMatrix(&(nsc->inversed_sqrt_ddg));
  FreeMatrix(&(nsc->l_norm));
  if (nsc->goal == LNORM)
    return;
  FreeMatrix(&(nsc->eigen_values));
  FreeMatrix(&(nsc->eigen_vectors));
}

void CalculateNandD(const char file_name[], int *n, int *d) {
  /* calculate number of input data data_points and dimensionality */
  char c;
  FILE *input_file = fopen(file_name, "r");
  assert(input_file != NULL);
  (*n) = 0, (*d) = 0;
  while ((c = (char) fgetc(input_file)) != EOF) {
    if (c == '\n') {
      (*d) += 1;
      break;
    }
    if (c == ',')
      (*d) += 1;
  }
  rewind(input_file);
  while ((c = (char) fgetc(input_file)) != EOF)
    if (c == '\n')
      (*n) += 1;
  fclose(input_file);
}
void BuildDataPointsMatrix(const char file_name[], double *data_points, int n, int d) {
  double data;
  int i = 0;
  FILE *input_file = fopen(file_name, "r");
  assert(input_file != NULL);
  while (fscanf(input_file, "%lf,", &data) != EOF) {
    data_points[i] = data;
    ++i;
  }
  fclose(input_file);
}
double CalculateWeight(int i, int j, Nsc *nsc) {
  /* i and j are the data data_points we want to find their weight */
  int k;
  double result;
  double *vector_i, *vector_j;
  vector_i = calloc(nsc->d, sizeof(double));
  assert(vector_i != NULL);
  vector_j = calloc(nsc->d, sizeof(double));
  assert(vector_j != NULL);
  for (k = 0; k < nsc->d; ++k) {
    vector_i[k] = (nsc->matrix)[i * nsc->d + k]; /* coping the ith row from matrix */
    vector_j[k] = (nsc->matrix)[j * nsc->d + k]; /* coping the jth row from matrix */
  }
  result = exp(-0.5 * CalculateEuclideanDistance(vector_i, vector_j, nsc->d));
  free(vector_i);
  free(vector_j);
  return result;
}
void CalculateATag(const double a[], double a_tag[], Nsc *nsc) {
  /* Declarations */
  double *p, *p_transpose, *help;
  int n = nsc->n;
  /* Memory allocation */
  AllocateMatrix(&p_transpose, n, n);
  /* Matrix calculations */
  Transpose(p, p_transpose, n);
  MultiplyTwoMatrices(p_transpose, a, help, n);
  MultiplyTwoMatrices(help, p, a_tag, n);
  /* Memory de-allocation*/
  free(p_transpose);
  free(help);

}
void CalculateATagEfficient(const double a[], double a_tag[], Nsc *nsc) {
  int i = nsc->i_pivot, j = nsc->j_pivot, n = nsc->n;
  double c = nsc->c, s = nsc->s;
  int r;
  for (r = 0; r < n; ++r) {
    if (r != nsc->i_pivot && r != nsc->j_pivot) {
      a_tag[r * n + i] = c * a[r * n + i] - s * a[r * n + j];
      a_tag[r * n + j] = c * a[r * n + j] + s * a[r * n + i];
    }
  }
  a_tag[i * n + i] = c * c * a[i * n + i] + s * s * a[j * n + j] - 2 * s * c * a[i * n + j];
  a_tag[j * n + j] = s * s * a[i * n + i] + c * c * a[j * n + j] + 2 * s * c * a[i * n + j];
  a_tag[i * n + j] = 0;
}

/****** The Eigen-gap Heuristic for finding number of clusters - K
 * values[n] , vectors[n * n], new_vectors[n * n]*****************/
int FindK(Nsc *nsc, int k) {
  double *new_values, *new_vectors, maximum;
  int i, j, index, max_index;
  double max = 0;
  new_values = calloc(nsc->n, sizeof(double));
  new_vectors = calloc(nsc->n * nsc->n, sizeof(double));
  if (new_values == NULL || new_vectors == NULL) {
    return -1;
  }
  maximum = FindMax(nsc->eigen_values, nsc->n);
  for (i = 0; i < nsc->n; i++) {
    index = IndexOfMinValue(nsc->eigen_values, nsc->n);
    new_values[i] = nsc->eigen_values[index];
    for (j = 0; j < nsc->n; j++) {
      new_vectors[j * nsc->n + i] = nsc->eigen_vectors[j * nsc->n + index];
    }
    nsc->eigen_values[index] = maximum + 1;
  }
  for (i = 0; i < nsc->n; i++) {
    nsc->eigen_values[i] = new_values[i];
  }
  free(new_values);
  CopyMatrix(nsc->eigen_vectors, new_vectors, nsc->n, nsc->n);
  free(new_vectors);
  if (k == 0) {
    for (i = 0; i < floor(nsc->n / 2.0); i++) {
      if (max < fabs(nsc->eigen_values[i] - nsc->eigen_values[i + 1])) {
        max = fabs(nsc->eigen_values[i] - nsc->eigen_values[i + 1]);
        max_index = i;
      }
    }
    return max_index + 1;
  }
  return k;
}

/*
 * Math helper functions
 */
double CalculateEuclideanDistance(double vector_1[], double vector_2[], int d) {
  /***
   * calculate and return the standard Euclidean distance
   * as defined in the project requirements.
   */
  double sum_of_squares = 0;
  int i;
  for (i = 0; i < d; ++i)
    sum_of_squares += pow(vector_2[i] - vector_1[i], 2);
  return sqrt(sum_of_squares);
}
void SubTwoMatrices(const double matrix_1[], const double matrix_2[], double sub[], int n) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      sub[i * n + j] = matrix_1[i * n + j] - matrix_2[i * n + j];
    }
  }
}
void MultiplyTwoMatrices(const double matrix_1[], const double matrix_2[], double product[], int n) {
  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        product[i * n + j] += matrix_1[i * n + k] * matrix_2[k * n + j];
      }
    }
  }
}
void IdentityMatrix(double identity[], int n) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i == j) {
        identity[i * n + j] = 1.0;
      } else {
        identity[i * n + j] = 0.0;
      }
    }
  }
}
int CheckDiagonal(const double a[], int n) {
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      if (i != j)
        if (a[i * n + j] != 0)
          return 0;
    }
  }
  return 1;
}
/* this func return the index of the min value in a given array */
int IndexOfMinValue(const double *values, int n) {
  int i;
  double min;
  int min_index;
  min = values[0];
  min_index = 0;
  for (i = 0; i < n; i++) {
    if (min > values[i]) {
      min = values[i];
      min_index = i;
    }
  }
  return min_index;
}
/*this func calculates the max of a given array*/
double FindMax(const double *values, int n) {
  int i;
  double max;
  max = values[0];
  for (i = 0; i < n; i++) {
    if (max < values[i]) {
      max = values[i];
    }
  }
  return max;
}
void Transpose(const double a[], double transpose[], int n) {
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      transpose[j * n + i] = a[i * n + j];
    }
  }
}

/*this func calculates U matrix*/
double *UMatrix(Nsc *nsc, int k) {
  double *u;
  int i, j;
  u = calloc(nsc->n * k, sizeof(double));
  assert(u != NULL);
  for (i = 0; i < nsc->n; i++) {
    for (j = 0; j < k; j++) {
      u[i * nsc->n + j] = nsc->eigen_vectors[i * nsc->n + j];
    }
  }
  return u;
}

/*this func calculates T matrix*/
double *TMatrix(double *u, int n, int k) {
  double *t;
  int i, j;
  double sum;
  t = calloc(n * k, sizeof(double));
  if (t == NULL) {
    return NULL;
  }
  for (i = 0; i < n; i++) {
    sum = 0.0;
    for (j = 0; j < k; j++) {
      sum += pow(u[i * n + j], 2);
    }
    sum = pow(sum, 0.5);
    for (j = 0; j < k; j++) {
      if (sum != 0) {
        t[i * n + j] = u[i * n + j] / sum;
      } else {
        t[i * n + j] = u[i * n + j];
      }
    }
  }
  return t;
}

double *Convert2dto1d(double **matrix, int n, int d) {
  int i, j;
  double *matrix_1d = calloc(n * d, sizeof(double));
  assert(matrix_1d != NULL);
  for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
      matrix_1d[i * d + j] = matrix[i][j];
    }
  }
  return matrix_1d;
}

double **Convert1dto2d(const double *matrix, int n, int d) {
  int i, j;
  double **matrix_2d = calloc(n * d, sizeof(double));
  for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
      matrix_2d[i][j] = matrix[i * d + j];
    }
  }
  return matrix_2d;
}
