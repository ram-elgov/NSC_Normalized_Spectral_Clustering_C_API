/* a small library with that implements the requirements
 * for the normalized spectral clustering algorithm */
#include "stdio.h"
#include "stdlib.h"
#include "spkmeans.h"
#include "assert.h"
#include "string.h"

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
	  break;
	case DDG:
	  printf("Calculate and output the Diagonal Degree"
			 " Matrix as described in 1.1.2.\n");
	  break;
	case LNORM:
	  printf("Calculate and output the Normalized"
			 " Graph Laplacian as described in 1.1.3.\n");
	  break;
	case JACOBI:
	  printf("Calculate and output the eigenvalues"
			 " and eigenvectors as described in 1.2.1.\n");
	  break;
  }
  PrintMatrix(nsc->matrix, nsc->n, nsc->d);
  return 0;
}
void ConstructNsc(Nsc *nsc, FILE *input_file) {
  CalculateNandD(nsc, input_file);
  InitDataPointsMatrix(nsc, input_file);
}
void CalculateNandD(Nsc *nsc, FILE *input_file) {
  /* calculate number of input data points and dimensionality */
  char c;
  nsc->n = 0;
  nsc->d = 0;
  while ((c = (char)fgetc(input_file)) != EOF) {
    if (c == '\n') {
      ++(nsc->d);
      break;
    }
    if (c == ',')
      ++(nsc->d);
  }
  rewind(input_file);
  while ((c = (char)fgetc(input_file)) != EOF)
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
  while (fscanf(input_file, "%lf,", &c) != EOF&& (i < nsc->d * nsc->n)) {
    if ((c != '\n') && (c != ',')) {
      (nsc->matrix)[i] = (double) c;
      ++i;
    }
  }
}
void PrintMatrix(double *matrix, int rows, int columns) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < columns; j++)
      printf("%lf ", matrix[i * columns + j]);
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
