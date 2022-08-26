/* a small library with that implements the requirements
 * for the normalized spectral clustering algorithm */
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
  FILE *input_file;
  /* Goal is an enum for the supported goals */
  Goal user_goal;
  double a[9] = {3,2,4,2,0,2,4,2,3};
  double *p;
  Nsc *nsc = malloc(sizeof(Nsc));
  assert(nsc != NULL);
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
  /* parse the file */
  input_file = fopen(argv[2], "r");
  assert(input_file != NULL);
  ConstructNsc(nsc, input_file, 0.00001);
  fclose(input_file);
  /* preform the required calculation */
  switch (user_goal) {
    case WAM:
      printf("Calculate and output the Weighted"
             " Adjacency Matrix as described in 1.1.1.\n");
      CalculateWeightedAdjacencyMatrix(nsc);
      PrintMatrix(nsc->wam, nsc->n, nsc->n);
      break;
    case DDG:
      printf("Calculate and output the Diagonal Degree"
             " Matrix as described in 1.1.2.\n");
      CalculateWeightedAdjacencyMatrix(nsc);
      CalculateDiagonalDegreeMatrix(nsc);
      PrintMatrix(nsc->ddg, nsc->n, nsc->n);
      break;
    case LNORM:
      CalculateNormalizedGraphLaplacian(nsc);
      printf("lnorm:\n");
      PrintMatrix(nsc->l_norm, nsc->n, nsc->n);
      break;
    case JACOBI:
      printf("Calculate and output the eigenvalues"
             " and eigenvectors as described in 1.2.1.\n");
      CalculateWeightedAdjacencyMatrix(nsc);
      printf("wam:\n");
      PrintMatrix(nsc->wam, nsc->n, nsc->n);
      CalculateDiagonalDegreeMatrix(nsc);
      printf("ddg:\n");
      PrintMatrix(nsc->ddg, nsc->n, nsc->n);
      InversedSqrtDiagonalDegreeMatrix(nsc);
      printf("Inversed sqrt ddg:\n");
      PrintMatrix(nsc->ddg, nsc->n, nsc->n);
      CalculateNormalizedGraphLaplacian(nsc);
      printf("lnorm:\n");
      PrintMatrix(nsc->l_norm, nsc->n, nsc->n);
      printf("Jacobi:\n");
      CalculateJacobi(nsc);
//      PrintMatrixJacobi(nsc);
      printf("\n");
      PrintMatrixJacobi(nsc);
//      PrintMatrix(nsc->eigen_vectors, nsc->n, nsc->n);
//      printf("a:\n");
//      PrintMatrix(a,3,3);
//      printf("p:\n");
//      nsc->n = 3;
//      p = Pmatrix(a, 3, nsc);
//      PrintMatrix(p,3,3);
//      printf("\n");
//      printf("a':\n");
//      PrintMatrix(CalculateATag(a,p,nsc),3,3);
      break;
  }
  DestructNsc(nsc);
  return 0;
}
void PrintMatrix(double matrix[], int rows, int columns) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < columns; j++) {
      printf("%.3f", matrix[i * rows + j]); /* matrix[i][j]*/
      if (j != columns - 1)
        printf(",");
    }
    printf("\n");
  }
}

void PrintMatrixJacobi(Nsc *nsc) {
  int i;
  int n = nsc->n;
  for (i = 0; i < n; i++) {
    if (nsc->eigen_values[i] > -0.00005 && nsc->eigen_values[i] < 0) {
      if (i < n - 1) {
        printf("%.3f%s", fabs(nsc->eigen_values[i]), ",");
      } else {
        printf("%.3f\n", fabs(nsc->eigen_values[i]));
      }
    } else {
      if (i < n - 1) {
        printf("%.3f%s", nsc->eigen_values[i], ",");
      } else {
        printf("%.3f\n", nsc->eigen_values[i]);
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


/*
 * library functions implementation
 */

void CalculateWeightedAdjacencyMatrix(Nsc *nsc) {
  double w_ij;
  int i, j;
  /* calculates the Weighted Adjacency Matrix.
   * https://moodle.tau.ac.il/mod/forum/discuss.php?d=127889
   * - use standard euclidean norm as defined in the project specification */
  nsc->wam = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->wam != NULL);
  /* we do not allow self loops, so we set w_ii = 0 for all i’s */
  for (i = 0; i < nsc->n; ++i) {
    (nsc->wam)[i * nsc->d + i] = 0; /* wam[i][i] = 0 */
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
  double val;
  int i, j;
  nsc->ddg = calloc(nsc->n * nsc->n , sizeof(double));
  assert(nsc->ddg != NULL);
  for(i = 0; i < nsc->n; i++) {
      val = 0;
      for (j = 0; j < nsc->n; j++) {
        val += (nsc->wam)[i * nsc->n + j];
      }
    (nsc->ddg)[i * nsc->n + i] = val;
  }
}
void CalculateNormalizedGraphLaplacian(Nsc *nsc) {
  double *identity;
  double *inversed_dw;
  double *inversed_dw_inversed_d;
  CalculateWeightedAdjacencyMatrix(nsc);
  CalculateDiagonalDegreeMatrix(nsc);
  InversedSqrtDiagonalDegreeMatrix(nsc);
  identity = IdentityMatrix(nsc->n);
  assert(identity != NULL);
  InversedSqrtDiagonalDegreeMatrix(nsc);
  inversedD_W = MultiplyTwoMatrices(nsc->ddg, nsc->wam, nsc->n);
  assert(inversedD_W != NULL);
  inversedD_W_inversedD = MultiplyTwoMatrices(inversedD_W, nsc->ddg, nsc->n);
  assert(inversedD_W_inversedD != NULL);
  nsc->l_norm = SubTwoMatrices(identity, inversedD_W_inversedD, nsc->n);
  assert(nsc->l_norm != NULL);
  free(identity);
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
double* CalculateJacobi(double Symmetric_matrix[], int n) {
  double *p, *a_tag, *a, *v;
  double convergence;
  int num_iteration, i;
  num_iteration = 0;
  /* on a_help we do the matrix operation instead of
   * overwriting l_norm */
  a = malloc(n * n * sizeof(double));
  assert(a != NULL);
  CopyMatrix(a, nsc->l_norm, n);
  /*  */
  a_tag = malloc(n * n * sizeof(double));
  assert(a_tag != NULL);
  CopyMatrix(a_tag, a, n);
  v = IdentityMatrix(n);
  assert(v != NULL);
  while(num_iteration < 100){
    p = Pmatrix(a, n, nsc);
    assert(p != NULL);
    a_tag = CalculateATag(a,p,nsc);
    assert(a_tag != NULL);
    v = MultiplyTwoMatrices(v, p, n);
    assert(v != NULL);
    convergence = (Off(a, n) - Off(a_tag, n));
    if(convergence <= nsc->epsilon){
      CopyMatrix(a, a_tag, n);
      break;
      assert(a != NULL);
    }
    CopyMatrix(a, a_tag, n);
    assert(a != NULL);
    num_iteration++;
    CopyMatrix(nsc->eigen_vectors, v, n);
  }
  for (i = 0; i < nsc->n; ++i) {
    nsc->eigen_values[i] = a[i * n + i];
  }
  free(a);
  free(a_tag);
  free(p);
}



/*
 * API helper functions
 */

void InversedSqrtDiagonalDegreeMatrix(Nsc *nsc){
  double val;
  int i;
  for(i = 0; i < nsc->n; i++) {
    val = (nsc->ddg)[i * nsc->n + i];
    (nsc->ddg)[i * nsc->n + i] = 1 / (sqrt(val));
  }
}

double *SubTwoMatrices(double matrix_1[], double matrix_2[], int n){
  double *sub = calloc(n*n, sizeof(double));
  int i, j;
  assert(sub != NULL);
  for(i = 0; i < n; i++){
      for(j = 0; j  < n; j++){
        sub[i * n + j] = matrix_1[i * n + j] - matrix_2[i * n + j];
      }
  }
  return sub;
}

double *MultiplyTwoMatrices(double matrix_1[], double matrix_2[], int n){
  double *multiply = calloc(n * n, sizeof(double));
  assert(multiply != NULL);
  int i,j,k;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      for(k = 0; k < n; k++){
        multiply[i * n + j] += matrix_1[i * n + k] * matrix_2[k * n + j];
      }
    }
  }
  return multiply;
}

double *IdentityMatrix(int n){
  double *identity = calloc(n * n, sizeof(double));
  assert(identity != NULL);
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(i == j){
        identity[i * n + j] = 1.0;
      } else {
        identity[i * n + j] = 0.0;
      }
    }
  }
  return identity;
}

void CalculateNormalizedGraphLaplacian(Nsc *nsc) {
  /* nsc->l_norm = SubTwoMatrices(IdentityMatrix(nsc),
                 MultiplyTwoMatrices(
                     MultiplyTwoMatrices(nsc->ddg, nsc->wam,nsc),
                     nsc->ddg,nsc),nsc); */
  double *identity;
  double *inversedD_W;
  double *inversedD_W_inversedD;
  identity = IdentityMatrix(nsc->n);
  assert(identity != NULL);
  InversedSqrtDiagonalDegreeMatrix(nsc);
  inversedD_W = MultiplyTwoMatrices(nsc->ddg, nsc->wam, nsc->n);
  assert(inversedD_W != NULL);
  inversedD_W_inversedD = MultiplyTwoMatrices(inversedD_W, nsc->ddg, nsc->n);
  assert(inversedD_W_inversedD != NULL);
  nsc->l_norm = SubTwoMatrices(identity, inversedD_W_inversedD, nsc->n);
  assert(nsc->l_norm != NULL);
  free(identity);
}
void ConstructNsc(Nsc *nsc, FILE *input_file, double epsilon) {
  nsc->epsilon = epsilon;
  CalculateNandD(nsc, input_file);
  InitDataPointsMatrix(nsc, input_file);
  nsc->eigen_values = malloc(nsc->n * sizeof(double));
  assert(nsc->eigen_values != NULL);
  nsc->eigen_vectors = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->eigen_vectors != NULL);
  nsc->ddg = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->ddg != NULL);
  nsc->inversed_sqrt_ddg = malloc(nsc->n * nsc->n * sizeof(double));
  assert(nsc->inversed_sqrt_ddg != NULL);
}
void DestructNsc(Nsc *nsc) {
  free(nsc->matrix);
  free(nsc->wam);
  free(nsc->ddg);
  free(nsc->inversed_sqrt_ddg);
  free(nsc->l_norm);
  free(nsc->eigen_values);
  free(nsc->eigen_vectors);
  free(nsc);
}
void CalculateNandD(Nsc *nsc, FILE *input_file) {
  /* calculate number of input data data_points and dimensionality */
  char c;
  nsc->n = 0;
  nsc->d = 0;
  while ((c = (char) fgetc(input_file)) != EOF) {
    if (c == '\n') {
      ++(nsc->d);
      break;
    }
    if (c == ',')
      ++(nsc->d);
  }
  rewind(input_file);
  while ((c = (char) fgetc(input_file)) != EOF)
    if (c == '\n')
      ++(nsc->n);
  rewind(input_file);
}
void InitDataPointsMatrix(Nsc *nsc, FILE *input_file) {
  double c;
  int i;
  /* initialize a data point matrix inside nsc */
  nsc->matrix = malloc((nsc->n * nsc->d) * sizeof(double));
  assert(nsc->matrix != NULL);
  i = 0;
  while (fscanf(input_file, "%lf,", &c) != EOF && (i < nsc->d * nsc->n)) {
    if ((c != '\n') && (c != ',')) {
      (nsc->matrix)[i] = (double) c;
      ++i;
    }
  }
}
double CalculateWeight(int i, int j, Nsc *nsc) {
  /* i and j are the data data_points we want to find their weight */
  int k;
  double result;
  double *vector_i, *vector_j;
  vector_i = malloc(nsc->d * sizeof(double));
  assert(vector_i != NULL);
  vector_j = malloc(nsc->d * sizeof(double));
  assert(vector_j != NULL);
  for (k = 0; k < nsc->d; ++k) {
    vector_i[k] = (nsc->matrix)[i * nsc->d + k]; /* coping the ith row from matrix */
    vector_j[k] = (nsc->matrix)[j * nsc->d + k]; /* coping the jth row from matrix */
  }
  result = exp(-0.5 * CalculateEuclideanDistance(vector_i, vector_j,nsc->d));
  free(vector_i);
  free(vector_j);
  return result;
}
double* CalculateATag(double a[], double p[], Nsc *nsc) {
  /* the commented code is a' calculation using standard calculation. */
  double *a_tag, *p_transpose, *help;
  int n = nsc->n;
  p_transpose = Transpose(p, n);

  assert(p_transpose != NULL);
  help = MultiplyTwoMatrices(p_transpose, a, n);
  assert(help != NULL);
  a_tag = MultiplyTwoMatrices(help, p, n);
  assert(a_tag != NULL);
  free(p_transpose);
  free(help);
  return a_tag;

}
double* CalculateATagEfficient(double a[], Nsc *nsc) {
  /* the commented code is a' calculation using efficient calculation. */
  int i = nsc->i_max, j = nsc->j_max, n = nsc->n;
  double c = nsc->c, s = nsc->s;
  double *a_tag;
  a_tag = calloc(n * n, sizeof(double));
  assert(a_tag != NULL);
  int r;
  for (r = 0; r < n; ++r) {
    if(r != nsc->i_max && r != nsc->j_max) {
      a_tag[r * n + i] = c * a[r * n + i] - s * a[r * n + j];
      a_tag[r * n + j] = c * a[r * n + j] + s * a[r * n + i];
    }
  }
  a_tag[i * n + i] = c * c * a[i * n + i] + s * s * a[j * n + j] - 2 * s * c * a[i * n + j];
  a_tag[j * n + j] = s * s * a[i * n + i] + c * c * a[j * n + j] + 2 * s * c * a[i * n + j];
  assert((c*c-s*s)*a[i * n + j] + s*c*(a[i * n + i]-a[j * n + j]) == 0);
  a_tag[i * n + j] = 0;
  CopyMatrix(a, a_tag, n);
  return a_tag;
}

/****** The Eigen-gap Heuristic for finding number of clusters - K
 * values[n] , vectors[n * n], new_vectors[n * n]*****************/
int FindK(double *values, const double *vectors, double *new_vectors, int n, int k) {
  double *new_values, maximum;
  int i, j, index, max_index;
  double max = 0;
  new_values = calloc(n, sizeof(double));
  if (new_values == NULL) {
    return -1;
  }
  maximum = FindMax(values, n);
  for (i = 0; i < n; i++) {
    index = IndexOfMinValue(values, n);
    new_values[i] = values[index];
    for (j = 0; j < n; j++) {
      new_vectors[j * n + i] = vectors[j * n + index];
    }
    values[index] = maximum + 1;
  }
  for (i = 0; i < n; i++) {
    values[i] = new_values[i];
  }
  free(new_values);
  if (k == 0) {
    for (i = 0; i < floor(n / 2); i++) {
      if (max < fabs(new_values[i] - new_values[i + 1])) {
        max = fabs(new_values[i] - new_values[i + 1]);
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
double *SubTwoMatrices(const double matrix_1[], const double matrix_2[], int n){
  double *sub = calloc(n*n, sizeof(double));
  int i, j;
  assert(sub != NULL);
  for(i = 0; i < n; i++){
    for(j = 0; j  < n; j++){
      sub[i * n + j] = matrix_1[i * n + j] - matrix_2[i * n + j];
    }
  }
  return sub;
}
double *MultiplyTwoMatrices(const double matrix_1[], const double matrix_2[], int n){
  double *multiply = calloc(n * n, sizeof(double));
  assert(multiply != NULL);
  int i,j,k;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      for(k = 0; k < n; k++){
        multiply[i * n + j] += matrix_1[i * n + k] * matrix_2[k * n + j];
      }
    }
  }
  return multiply;
}
double *IdentityMatrix(int n){
  double *identity = calloc(n * n, sizeof(double));
  assert(identity != NULL);
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(i == j){
        identity[i * n + j] = 1.0;
      } else {
        identity[i * n + j] = 0.0;
      }
    }
  }
  return identity;
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
double* Transpose(double a[], int n) {
  int i, j;
  double *transpose = malloc(n * n * sizeof(double));
  assert(transpose != NULL);
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      transpose[j * n + i] = a[i * n + j];
    }
  }
  return transpose;
}

/*this func calculates U matrix*/
double *UMatrix(int n, int k, const double *new_vectors) {
    double *u;
    int i, j;
    u = calloc(n * k, sizeof(double));
    if (u == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            u[i * n + j] = new_vectors[i * n + j];
        }
    }
    return u;
}

/*this func calculates T matrix*/
double **TMatrix(double *u, int n, int k) {
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
