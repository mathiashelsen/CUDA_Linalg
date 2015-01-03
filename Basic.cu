#include "Basic.h"
#include "Basic.cuh"

__global__ void MatAdd(Matrix A, Matrix B, Matrix C)
{
    if( A.rows != B.rows)
	return;
    if( A.cols != B.cols)
	return;

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y; 
    if( (col < C.cols) && (row < C.rows) )
	C.elems[row + col*C.rows] = A.elems[row + col*A.rows] + B.elems[row + col*B.rows];
}

void GPUMatAdd(Matrix A, Matrix B, Matrix C)
{
    int sizeA = A.rows * A.cols * sizeof(float);

    Matrix a(A.rows, A.cols, MemoryLocationGPU), b(B.rows, B.cols, MemoryLocationGPU), c(C.rows, C.cols, MemoryLocationGPU);

    cudaMemcpy( a.elems, A.elems, sizeA, cudaMemcpyHostToDevice );
    cudaMemcpy( b.elems, B.elems, sizeA, cudaMemcpyHostToDevice );

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(A.cols/threadsPerBlock.x + 1, A.rows/threadsPerBlock.y + 1);

    MatAdd<<<numBlocks, threadsPerBlock >>>(a, b, c);
    
    cudaMemcpy( C.elems, c.elems, sizeA, cudaMemcpyDeviceToHost ); 

    a.free();
    b.free();
    c.free();
}

void CPUMatAdd(Matrix A, Matrix B, Matrix C)
{
    int i = 0, j = 0; 
    for( i = 0; i < A.cols; i++ )
    {
	for( j = 0; j < A.rows; j++ )
	{
	    C.elems[j + i*A.rows] = A.elems[j + i*A.rows] + B.elems[j + i*A.rows];
	}
    }
}
