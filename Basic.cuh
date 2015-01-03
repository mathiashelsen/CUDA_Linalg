#ifndef _BASIC_CUH
#define _BASIC_CUH

#include <cuda.h>
#include "Matrix.h"

__global__ void MatAdd(Matrix A, Matrix B, Matrix C);

#endif
