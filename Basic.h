#ifndef _BASIC_H_
#define _BASIC_H_

#include <cuda.h>

#include <stdio.h>

#include "Matrix.h"

void GPUMatAdd(Matrix A, Matrix B, Matrix C);
void CPUMatAdd(Matrix A, Matrix B, Matrix C);

#endif
