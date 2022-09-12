#ifndef TEST_SPKMEANS_LIB__SPKMEANS_H_
#define TEST_SPKMEANS_LIB__SPKMEANS_H_
#include "stdio.h"
/**
 *
 */
typedef enum {
  FALSE,
  TRUE
} bool;
/**
 *
 */
typedef enum {
  WAM,
  DDG,
  LNORM,
  JACOBI,
  FIT
} Goal;
/**
 *
 */
typedef struct normalized_spectral_clustering {
  double *matrix, *ddg, *inversed_sqrt_ddg, *wam, *l_norm,
      *eigen_vectors, *eigen_values;
  int n, d, i_pivot, j_pivot;
  double s, c, epsilon;
  Goal goal;
} Nsc;

/* standalone client */
/**
 *
 */
void InvalidInput();
/**
 *
 */
void GeneralError();
/**
 *
 * @param matrix
 * @param rows
 * @param d
 */
void PrintMatrix(const double *matrix, int rows, int d);
/**
 *
 * @param matrix
 * @param n
 * @param d
 */
void AllocateMatrix(double **matrix, int n, int d);
/**
 *
 * @param matrix
 */
void FreeMatrix(double **matrix);
/**
 *
 * @param nsc
 */
void ChooseGoal(Nsc *nsc);

/* the spectral clustering API (library functions) */
/**
 *
 * @param nsc
 */
void CalculateWeightedAdjacencyMatrix(Nsc *nsc);
/**
 *
 * @param nsc
 */
void CalculateDiagonalDegreeMatrix(Nsc *nsc);
/**
 *
 * @param nsc
 */
void InversedSqrtDiagonalDegreeMatrix(Nsc *nsc);
/**
 *
 * @param nsc
 */
void CalculateNormalizedGraphLaplacian(Nsc *nsc);
/**
 *
 * @param nsc
 */
void CalculateJacobi(Nsc *nsc);

/* API helper functions */
/**
 *
 * @param nsc
 * @param data_points
 * @param n
 * @param d
 * @param goal
 */
void ConstructNsc(Nsc *nsc, double *data_points, int n, int d, Goal
goal);
/**
 *
 * @param nsc
 */
void DestructNsc(Nsc *nsc);
/**
 *
 * @param file_name
 * @param n
 * @param d
 */
void CalculateNandD(const char file_name[], int *n, int *d);
/**
 *
 * @param file_name
 * @param data_points
 */
void BuildDataPointsMatrix(const char file_name[],
                           double *data_points);
/**
 *
 * @param a
 * @param p
 * @param n
 * @param nsc
 */
void CalculateRotationMatrix(const double a[], double p[], int n, Nsc *nsc);
/**
 *
 * @param a
 * @param a_tag
 * @param p
 * @param v
 * @param n
 * @param nsc
 */
void RunJacobiCalculations(double a[], double a_tag[], double p[], double
v[], int n, Nsc *nsc);
/**
 *
 * @param a
 * @param n
 * @param pivot
 * @param i_pivot
 * @param j_pivot
 */
void FindPivot(const double a[], int n,
               double *pivot, int *i_pivot, int *j_pivot);
/**
 *
 * @param theta
 * @return
 */
int Sign(double theta);
/**
 *
 * @param a
 * @param n
 * @return
 */
double Off(double a[], int n);
/**
 *
 * @param a
 * @param b
 * @param n
 * @param d
 */
void CopyMatrix(double a[], const double b[], int n, int d);
/**
 *
 * @param i
 * @param j
 * @param nsc
 * @return
 */
double CalculateWeight(int i, int j, Nsc *nsc);
/**
 *
 * @param a
 * @param a_tag
 * @param nsc
 */
void CalculateAPrimeEfficient(const double a[], double a_tag[], Nsc *nsc);
/**
 *
 * @param nsc
 * @param k
 * @return
 */
int FindK(Nsc *nsc, int k);
/**
 *
 * @param nsc
 * @param u
 * @param k
 */
void CalculateUMatrix(Nsc *nsc, double *u, int k);
/**
 *
 * @param u
 * @param t
 * @param n
 * @param k
 */
void CalculateTMatrix(double *u, double *t, int n, int k);

/* Math helper functions */
/**
 *
 * @param vector_1
 * @param vector_2
 * @param d
 * @return
 */
double CalculateEuclideanDistance(double vector_1[], double vector_2[], int d);
/**
 *
 * @param matrix_1
 * @param matrix_2
 * @param sub
 * @param n
 */
void SubTwoMatrices(const double matrix_1[],
                    const double matrix_2[],
                    double sub[],
                    int n);
/**
 *
 * @param matrix_1
 * @param matrix_2
 * @param product
 * @param n
 */
void MultiplyTwoMatrices(const double matrix_1[],
                         const double matrix_2[],
                         double product[],
                         int n);
void IdentityMatrix(double identity[], int n);
/**
 *
 * @param a
 * @param n
 * @return
 */
bool IsDiagonal(const double a[], int n);

/**
 *
 * @param values
 * @param n
 * @return
 */
int IndexOfMaxValue(const double *values, int n);
/**
 *
 * @param values
 * @param n
 * @return
 */
double FindMin(const double *values, int n);
#endif