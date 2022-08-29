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
      PrintMatrix(nsc->inversed_sqrt_ddg, nsc->n, nsc->n);
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
    if (nsc->eigen_values[i] > -0.00001 && nsc->eigen_values[i] < 0) {
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
  inversed_dw = MultiplyTwoMatrices(nsc->inversed_sqrt_ddg, nsc->wam, nsc->n);
  assert(inversed_dw != NULL);
  inversed_dw_inversed_d = MultiplyTwoMatrices(inversed_dw, nsc->inversed_sqrt_ddg, nsc->n);
  assert(inversed_dw_inversed_d != NULL);
  nsc->l_norm = SubTwoMatrices(identity, inversed_dw_inversed_d, nsc->n);
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
void CalculateJacobi(Nsc *nsc) {
  double *p, *a_tag, *a, *v;
  double convergence;
  int num_iteration, i, n = nsc->n;
  num_iteration = 0;
  /* on a_help we do the matrix operation instead of
   * overwriting l_norm */
  a = calloc(n * n, sizeof(double));
  assert(a != NULL);
  CopyMatrix(a, nsc->l_norm, n, n);
  /*  */
  a_tag = calloc(n * n, sizeof(double));
  assert(a_tag != NULL);
  CopyMatrix(a_tag, a, n,n);
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
      CopyMatrix(a, a_tag, n,n);
      break;
      assert(a != NULL);
    }
    CopyMatrix(a, a_tag, n,n);
    assert(a != NULL);
    num_iteration++;
    CopyMatrix(nsc->eigen_vectors, v, n,n);
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
    (nsc->inversed_sqrt_ddg)[i * nsc->n + i] = 1 / (sqrt(val));
  }
}
double *Pmatrix(double a[], int n, Nsc *nsc){
  int i,j,i_max,j_max;
  double c,s,teta,t,max;
  double *p;
  max = a[1];
  i_max = 0;
  j_max = 1;
  p = IdentityMatrix(n);
  if(p == NULL){
    return NULL;
  }
  for(i = 0; i < n; i++){
    for(j = i + 1; j < n; j++){
      if(fabs(a[i * n + j]) > fabs(max)){
        max = a[i * n + j];
        i_max = i;
        j_max = j;
      }
    }
  }
  teta = (a[j_max * n + j_max] - a[i_max * n + i_max]) / (2 * max);
  if (teta >= 0) {
    t = 1 / (fabs(teta) + sqrt(1 + teta * teta));
  } else {
    t = -1 / (fabs(teta) + sqrt(1 + teta * teta));
  }
  c = 1 / (sqrt(t * t + 1));
  s = t * c;
  p[i_max * n + i_max] = c;
  p[j_max * n + j_max] = c;
  p[i_max * n + j_max] = s;
  p[j_max * n + i_max] = -1 * s;
  nsc->s = s;
  nsc->c = c;
  nsc->i_max = i_max;
  nsc->j_max = j_max;
  return p;
}
double Off(double a[], int n){
  double off = 0.0;
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(i != j){
        off += pow(a[i * n + j], 2);
      }
    }
  }
  return off;
}
void CopyMatrix(double a[], const double b[], int n, int d){
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < d; j++){
      a[i * n + j] = b[i * n + j];
    }
  }
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
  CopyMatrix(a, a_tag, n,n);
  return a_tag;
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
    for (i = 0; i < floor(nsc->n / 2); i++) {
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
double *Transpose(double a[], int n) {
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
double* TMatrix(double *u, int n, int k) {
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
