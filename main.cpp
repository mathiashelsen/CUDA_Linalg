#include <assert.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>

#include "Basic.h"
#include "Matrix.h"

using namespace std;

uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

int main(int argc, char **argv)
{
    uint64_t start = 0, stop = 0;
    int N = 1024;

    Matrix *A = new Matrix(N, N, MemoryLocationCPU);
    Matrix *B = new Matrix(N, N, MemoryLocationCPU);
    Matrix *C = new Matrix(N, N, MemoryLocationCPU);
    Matrix *D = new Matrix(N, N, MemoryLocationCPU);
    Matrix *E = new Matrix(N, N, MemoryLocationCPU);

    boost::random::mt19937 rng; 
    boost::random::uniform_01<> dist;
    for(int i = 0; i < A->cols; i++)
    {
	for(int j = 0; j < A->rows; j++)
	{
	    A->elems[i*A->rows + j] = dist(rng);
	    B->elems[i*A->rows + j] = dist(rng);
	}
    }

    std::cout << "Starting CPU Matrix Multiplication" << std::endl;
    start = GetTimeStamp();
    CPUMatMulNaive(*A, *B, *C);
    stop = GetTimeStamp();
    std::cout << "Done on CPU, runtime = " << (stop-start)<< " s" << std::endl;


    std::cout << "Starting GPU Matrix Multiplication" << std::endl;
    start = GetTimeStamp();
    GPUMatMul(*A, *B, *D);
    stop = GetTimeStamp();
    std::cout << "Done on GPU, runtime = " << stop-start << " us" << std::endl;

    std::cout << "Starting Naive GPU Matrix Multiplication" << std::endl;
    start = GetTimeStamp();
    GPUMatMulNaive(*A, *B, *E);
    stop = GetTimeStamp();
    std::cout << "Done on GPU, runtime = " << (stop-start)<< " s" << std::endl;

    std::cout << "Comparing both results" << std::endl;

    for( int i = 0; i < A->cols; i++ )
    {
	for( int j = 0; j < A->rows; j++ )
	{
	    /*printf( "%e, %e, %e\n", 
		C->elems[i*C->rows + j], 
		D->elems[i*D->rows + j], 
		C->elems[i*C->rows + j] - D->elems[i*D->rows + j] );*/
	    assert( fabs(C->elems[j*C->cols + i] - D->elems[j*D->cols + i]) < 1.0e-3 );
	}
    }
    std::cout << "Assertion did not fail." << std::endl;
    A->free();
    B->free();
    C->free();
    D->free();
    delete A, B, C, D;
};
