/* a small library with that implements the requirements
 * for the normalized spectral clustering algorithm */
#include "stdio.h"
#include "stdlib.h"
#include "spkmeans.h"
#include "assert.h"
#include "string.h"
#include "math.h"
/* standalone client */
int main(int argc, char **argv) {
  FILE *input_file;
  /* Goal is an enum for the supported goals */
  Goal user_goal;
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
  ConstructNsc(nsc, input_file);
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
      printf("Calculate and output the Normalized"
             " Graph Laplacian as described in 1.1.3.\n");
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
      PrintMatrix(nsc->eigen_values, 1, nsc->n);
      PrintMatrix(nsc->eigen_vectors, nsc->n, nsc->n);
      break;
  }
  DestructNsc(nsc);
  return 0;
}
void PrintMatrix(double *matrix, int rows, int columns) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < columns; j++) {
      printf("%.3f", matrix[i * columns + j]); /* matrix[i][j]*/
      if (j != columns - 1)
        printf(",");
    }
    printf("\n");
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
  double *P, *Aprime, *Ahelp, *V, *Vhelp;
  double convergence, epsilon;
  int num_iteration, i, j, n = nsc->n;
  num_iteration = 0;
  epsilon = pow(10, -5);
  CopyMatrix(Ahelp, nsc->l_norm, n);
  assert(Ahelp != NULL);
  CopyMatrix(Aprime, Ahelp, n);
  assert(Aprime != NULL);
  V = IdentityMatrix(n);
  assert(V != NULL);
  while(num_iteration < 100){
    P = Pmatrix(n, nsc);
    assert(P != NULL);
    CalculateATag(Aprime,nsc);
    assert(Aprime != NULL);
//    CopyMatrix(Vhelp, V, n);
//    assert(Vhelp != NULL);
//    free(V);
    V = MultiplyTwoMatrices(V, P, n);
    assert(V != NULL);
    convergence = (Off(Ahelp, n) - Off(Aprime, n));
    if(convergence <= epsilon){
      free(Ahelp);
      CopyMatrix(Ahelp, Aprime, n);
      assert(Ahelp != NULL);
      free(Vhelp);
      free(Aprime);
      free(P);
      break;
    }
    CopyMatrix(Ahelp, Aprime, n);
    assert(Ahelp != NULL);
    num_iteration++;
    CopyMatrix(nsc->eigen_vectors, Vhelp, n);
  }
  for (i = 0; i < nsc->n; ++i) {
    nsc->eigen_values[i] = Ahelp[i * n + i];
  }
  free(Ahelp);
  free(Vhelp);
  free(Aprime);
  free(P);
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
void ConstructNsc(Nsc *nsc, FILE *input_file) {
  CalculateNandD(nsc, input_file);
  InitDataPointsMatrix(nsc, input_file);
}
void DestructNsc(Nsc *nsc) {
  free(nsc->matrix);
  free(nsc->wam);
  free(nsc->ddg);
  free(nsc->l_norm);
  free(nsc->eigen_values);
  free(nsc->eigen_vectors);
  free(nsc);
}
void CalculateNandD(Nsc *nsc, FILE *input_file) {
  /* calculate number of input data points and dimensionality */
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
  /* i and j are the data points we want to find their weight */
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
void CalculateATag(double a[], Nsc *nsc) {
  int i = nsc->i_max, j = nsc->j_max, n = nsc->n;
  double c = nsc->c, s = nsc->s;
  double *a_tag;
  a_tag = malloc(n * n * sizeof(double));
  assert(a_tag != NULL);
  int r;
  for (r = 0; r < n; ++r) {
    if(r != nsc->i_max && r != nsc->j_max) {
      a_tag[r * n + i] = c * a[r * n + i] - s * a[r * n + j];
      a_tag[r * n + j] = c * a[r * n + j] + s * a[r * n + i];
    }
  }
  a_tag[i * n + i] = c * c * a[i * n + i] - s * s * a[j * n + j] - 2 * s * c * a[i * n + j];
  a_tag[j * n + j] = c * c * a[i * n + i] - s * s * a[j * n + j] + 2 * s * c * a[i * n + j];
  a_tag[i * n + j] = 0;
  CopyMatrix(a, a_tag, n);
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
  int i = 0;
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