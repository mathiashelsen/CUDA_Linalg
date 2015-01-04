#include <assert.h>
#include <iostream>
#include <stdlib.h>

#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>

#include "Basic.h"
#include "Matrix.h"

using namespace std;

int main(int argc, char **argv)
{
    int N = 64;

    Matrix *A = new Matrix(N, N, MemoryLocationCPU);
    Matrix *B = new Matrix(N, N, MemoryLocationCPU);
    Matrix *C = new Matrix(N, N, MemoryLocationCPU);
    Matrix *D = new Matrix(N, N, MemoryLocationCPU);

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
    CPUMatMul(*A, *B, *C);
    std::cout << "Done on CPU" << std::endl;
    std::cout << "Starting GPU Matrix Multiplication" << std::endl;
    GPUMatMul(*A, *B, *D);
    std::cout << "Done on GPU" << std::endl;

    std::cout << "Comparing both results" << std::endl;

    for( int i = 0; i < A->cols; i++ )
    {
	for( int j = 0; j < A->rows; j++ )
	{
	    printf( "%e, %e, %e\n", 
		C->elems[i*C->rows + j], 
		D->elems[i*D->rows + j], 
		C->elems[i*C->rows + j] - D->elems[i*D->rows + j] );
	    assert( fabs(C->elems[i*C->rows + j] - D->elems[i*D->rows + j]) < 1.0e-5 );
	}
    }
    std::cout << "Assertion did not fail." << std::endl;
    A->free();
    B->free();
    C->free();
    D->free();
    delete A, B, C, D;
};
