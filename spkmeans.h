#ifndef SPECTRAL_CLUSTERING__SPKMEANS_H_
#define SPECTRAL_CLUSTERING__SPKMEANS_H_
/*
 * the spectral clustering API
 * */
typedef struct normalized_spectral_clustering {
  /**
   * n: number of data points
   * d: data point dimension
   * k: number of required clusters
   */
  double *matrix, ddm, wam, l_norm, t;
  int n, d, k;
} Nsc;
typedef enum {
  WAM,
  DDG,
  LNORM,
  JACOBI
} Goal;
void InvalidInput();
void GeneralError();

/* standalone client */
void ConstructNsc(Nsc *nsc, FILE *input);
void DestructNsc(Nsc *nsc);
void CalculateNandD(Nsc *nsc, FILE *input_file);
void InitDataPointsMatrix(Nsc *nsc, FILE *input_file);
void PrintMatrix(double *matrix, int rows, int columns);
#endif /* SPECTRAL_CLUSTERING__SPKMEANS_H_ */
