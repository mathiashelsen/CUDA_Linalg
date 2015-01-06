#include "Basic.h"
#include "Basic.cuh"

#define BLOCK_SIZE 16

__global__ void MatAddKernel(Matrix A, Matrix B, Matrix C)
{
    if( A.rows != B.rows)
	return;
    if( A.cols != B.cols)
	return;

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y; 
    if( (col < C.cols) && (row < C.rows) )
	C.elems[col + row*C.cols] = A.elems[col + row*A.cols] + B.elems[col + row*B.cols];
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
	    sum += A.elems[i + row*A.cols]*B.elems[i*B.cols + col];
	}
	C.elems[col + row*C.cols] = sum;
    }
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    int NSubBlocks = A.cols/BLOCK_SIZE;
    float sum = 0.0;

    int col = threadIdx.x + blockIdx.x*blockDim.x;
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int blockCol = threadIdx.x;
    int blockRow = threadIdx.y;

    // For each sub-matrix perform the calculations
    for( int block = 0; block < NSubBlocks; block++ )
    {
	__shared__ float As[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];

	As[blockCol + blockRow*BLOCK_SIZE] = A.elems[(blockCol + block*BLOCK_SIZE) + row*A.cols];
	// Read in transposed matrix
	Bs[blockCol + blockRow*BLOCK_SIZE] = B.elems[col + (blockRow + block*BLOCK_SIZE)*B.cols];

	__syncthreads();
	for(int i = 0; i < BLOCK_SIZE; i++)
	{
	    sum += As[i + blockRow*BLOCK_SIZE]*Bs[blockCol + i*BLOCK_SIZE];
	}

	__syncthreads();
    }
    C.elems[col + row*C.cols] = sum;
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
    for( i = 0; i < A.rows; i++ )
    {
	for( j = 0; j < A.cols; j++ )
	{
	    C.elems[j + i*A.cols] = A.elems[j + i*A.cols] + B.elems[j + i*A.cols];
	}
    }
}

void GPUMatMulNaive(Matrix A, Matrix B, Matrix C)
{
    Matrix a(A.rows, A.cols, MemoryLocationGPU);
    Matrix b(B.rows, B.cols, MemoryLocationGPU);
    Matrix c(A.rows, B.cols, MemoryLocationGPU);

    cudaMemcpy( a.elems, A.elems, A.rows*A.cols*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( b.elems, B.elems, B.rows*B.cols*sizeof(float), cudaMemcpyHostToDevice );

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(C.cols/threadsPerBlock.x, C.rows/threadsPerBlock.y);

    MatMulNaiveKernel<<<numBlocks, threadsPerBlock>>>(a, b, c);

    cudaMemcpy( C.elems, c.elems, C.rows*C.cols*sizeof(float), cudaMemcpyDeviceToHost);

    a.free();
    b.free();
    c.free();

}

void GPUMatMul(Matrix A, Matrix B, Matrix C)
{
    Matrix a(A.rows, A.cols, MemoryLocationGPU);
    Matrix b(B.rows, B.cols, MemoryLocationGPU);
    Matrix c(A.rows, B.cols, MemoryLocationGPU);

    cudaMemcpy( a.elems, A.elems, A.rows*A.cols*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( b.elems, B.elems, B.rows*B.cols*sizeof(float), cudaMemcpyHostToDevice );

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(C.cols/threadsPerBlock.x, C.rows/threadsPerBlock.y);

    MatMulKernel<<<numBlocks, threadsPerBlock>>>(a, b, c);

    cudaMemcpy( C.elems, c.elems, C.rows*C.cols*sizeof(float), cudaMemcpyDeviceToHost);

    a.free();
    b.free();
    c.free();
}

