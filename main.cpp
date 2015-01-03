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
    int N = atoi(argv[1]);

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

    CPUMatAdd(*A, *B, *C);
    GPUMatAdd(*A, *B, *D);

    for( int i = 0; i < A->cols; i++ )
    {
	for( int j = 0; j < A->rows; j++ )
	{
	    assert( C->elems[i*A->rows + j] == D->elems[i*A->rows + j] );
	}
    }
    std::cout << "Assertion did not fail." << std::endl;
    A->free();
    B->free();
    C->free();
    D->free();
    delete A, B, C, D;
};
