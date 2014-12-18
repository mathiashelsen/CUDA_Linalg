#include "CPUMatMul.h"

void CPUMatMul(Matrix *A, Matrix *B, Matrix *C)
{
    for(int i = 0; i < C->cols; i++)
    {
//	#pragma omp parallel for
	for(int j = 0; j < C->rows; j++)
	{
	    float sum = 0.0;
	    for(int k = 0; k < A->cols; k++)
	    {
		sum += A->elems[k*A->rows + j]*B->elems[i*B->rows + k];
	    }
	    C->elems[i*C->rows + j] = sum;
	}
//	#pragma omp barrier
    }
};
