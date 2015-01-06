#ifndef _BASIC_H_
#define _BASIC_H_

#include <cuda.h>

#include <stdio.h>

#include "Matrix.h"

void GPUMatAdd(Matrix A, Matrix B, Matrix C);
void CPUMatAdd(Matrix A, Matrix B, Matrix C);
void GPUMatMul(Matrix A, Matrix B, Matrix C);
void GPUMatMulNaive(Matrix A, Matrix B, Matrix C);

void CPUMatMul(Matrix A, Matrix B, Matrix C);
void CPUMatMulNaive(Matrix A, Matrix B, Matrix C);

#endif
