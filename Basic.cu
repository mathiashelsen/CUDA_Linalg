#include "Basic.h"
#include "Basic.cuh"

__global__ void MatAddKernel(Matrix A, Matrix B, Matrix C)
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

__global__ void MatMulNaiveKernel(Matrix A, Matrix B, Matrix C)
{
    if( A.cols != B.rows )
	return;
    if( A.rows != C.rows )
	return;
    if( B.cols != C.cols )
	return;

    int col = threadIdx.x + blockIdx.x*blockDim.x;
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    if( (col < C.cols) && (row < C.rows) )
    {
	int i = 0;
	float sum = 0.0;
	for( i = 0; i < A.cols; i++ )	
	{
	    sum += A.elems[row + i*A.rows]*B.elems[col*B.rows + i];
	}
	C.elems[row + col*C.rows] = sum;
    }
}

void GPUMatAdd(Matrix A, Matrix B, Matrix C)
{
    int sizeA = A.rows * A.cols * sizeof(float);

    Matrix a(A.rows, A.cols, MemoryLocationGPU), b(B.rows, B.cols, MemoryLocationGPU), c(C.rows, C.cols, MemoryLocationGPU);

    cudaMemcpy( a.elems, A.elems, sizeA, cudaMemcpyHostToDevice );
    cudaMemcpy( b.elems, B.elems, sizeA, cudaMemcpyHostToDevice );

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(A.cols/threadsPerBlock.x + 1, A.rows/threadsPerBlock.y + 1);

    MatAddKernel<<<numBlocks, threadsPerBlock >>>(a, b, c);
    
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

void GPUMatMul(Matrix A, Matrix B, Matrix C)
{
    Matrix a(A.rows, A.cols, MemoryLocationGPU);
    Matrix b(B.rows, B.cols, MemoryLocationGPU);
    Matrix c(A.rows, B.cols, MemoryLocationGPU);

    cudaMemcpy( a.elems, A.elems, A.rows*A.cols*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( b.elems, B.elems, B.rows*B.cols*sizeof(float), cudaMemcpyHostToDevice );

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(C.cols/threadsPerBlock.x + 1, C.rows/threadsPerBlock.y + 1);

    MatMulNaiveKernel<<<numBlocks, threadsPerBlock>>>(a, b, c);

    cudaMemcpy( C.elems, c.elems, C.rows*C.cols*sizeof(float), cudaMemcpyDeviceToHost);

    a.free();
    b.free();
    c.free();
}

void CPUMatMul(Matrix A, Matrix B, Matrix C)
{
    int row = 0, col = 0;
    for( col = 0; col < C.cols; col++ )
    {
	for( row = 0; row < C.rows; row++ )
	{
	    float sum = 0.0;
	    int i = 0;
	    for( i = 0; i < A.cols; i++ )
	    {
		sum += A.elems[i*A.rows + row]*B.elems[col*B.rows + i];
	    }
	    C.elems[col*C.rows + row] = sum;
	}
    }
}
