#ifndef _MATADD_H
#define _MATADD_H

#include <cuda.h>
#include "Matrix.h"

//__global__ void MatAdd(Matrix A, Matrix B, Matrix C);

void GPUMatAdd(Matrix A, Matrix B, Matrix C);
void CPUMatAdd(Matrix A, Matrix B, Matrix C);

#endif
