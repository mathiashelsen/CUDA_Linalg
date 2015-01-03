#ifndef _MATADD_CUH
#define _MATADD_CUH

#include <cuda.h>
#include "Matrix.h"

__global__ void MatAdd(Matrix A, Matrix B, Matrix C);

#endif
