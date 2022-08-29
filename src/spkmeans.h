#ifndef NSC___NORMALIZED_SPECTRAL_CLUSTERING_C_API_SRC_SPKMEANS_H_
#define NSC___NORMALIZED_SPECTRAL_CLUSTERING_C_API_SRC_SPKMEANS_H_
/*
 * the spectral clustering API (library functions)
 * */
typedef struct normalized_spectral_clustering {
  /**
   * n: number of data data_points
   * d: data point dimension
   * k: number of required clusters
   */
  double *matrix, *ddg,*inversed_sqrt_ddg, *wam, *l_norm, *eigen_vectors, *eigen_values;
  int n, d, i_max, j_max;
  double s, c, epsilon;
} Nsc;
typedef enum {
  WAM,
  DDG,
  LNORM,
  JACOBI
} Goal;
/* Calculate and output the Weighted Adjacency Matrix as described in 1.1.1. */
void CalculateWeightedAdjacencyMatrix(Nsc *nsc);
/* Calculate and output the Diagonal Degree Matrix as described in 1.1.2. */
void CalculateDiagonalDegreeMatrix(Nsc *nsc);
void InversedSqrtDiagonalDegreeMatrix(Nsc *nsc);
void CalculateNormalizedGraphLaplacian(Nsc *nsc);
void CalculateJacobi(Nsc *nsc);


/* API helper functions */
void ConstructNsc(Nsc *nsc, FILE *input, double epsilon);
void DestructNsc(Nsc *nsc);
void CalculateNandD(Nsc *nsc, FILE *input_file);
void InitDataPointsMatrix(Nsc *nsc, FILE *input_file);
/* Let w_ij represent the weight of the connection between v_i and v_j . Only if w ij > 0, we define an edge
between v_i and v_j. */
double *Pmatrix(double a[], int n, Nsc *nsc);
double Off(double a[], int n);
void CopyMatrix(double a[],const double b[], int n, int d);
double CalculateWeight(int i, int j, Nsc *nsc);
double* CalculateATag(double a[], double p[], Nsc *nsc);
double* CalculateATagEfficient(double a[], Nsc *nsc);
int FindK(Nsc *nsc, int k);
double *UMatrix(Nsc *nsc, int k);
double *TMatrix(double *u, int n, int k);

/* Math helper functions */
double CalculateEuclideanDistance(double vector_1[], double vector_2[], int d);
double *SubTwoMatrices(const double matrix_1[], const double matrix_2[], int n);
double *MultiplyTwoMatrices(const double matrix_1[], const double matrix_2[], int n);
double *IdentityMatrix(int n);
int CheckDiagonal(const double a[], int n);
int IndexOfMinValue(const double *values, int n);
double FindMax(const double *values, int n);
double *Transpose(double matrix[], int n);


/* standalone client */
void InvalidInput();
void GeneralError();
void PrintMatrix(double *matrix, int rows, int columns);
void PrintMatrixJacobi(Nsc *nsc);
#endif /* NSC___NORMALIZED_SPECTRAL_CLUSTERING_C_API_SRC_SPKMEANS_H_ */
