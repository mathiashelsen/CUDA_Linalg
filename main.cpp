#include <iostream>
#include <stdlib.h>
#include "Matrix.h"
#include "CPUMatMul.h"

#include <boost/random/uniform_01.hpp>
#include <boost/random.hpp>

using namespace std;

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);

    Matrix A;
    A.cols = N;
    A.rows = N;
    A.elems = new float[N*N];
    Matrix B;
    B.cols = N;
    B.rows = N;
    B.elems = new float[N*N];
    Matrix C;
    C.cols = N;
    C.rows = N;
    C.elems = new float[N*N];

    boost::random::mt19937 rng; 
    boost::random::uniform_01<> dist;
    for(int i = 0; i < A.cols; i++)
    {
	for(int j = 0; j < A.rows; j++)
	{
	    A.elems[i*A.rows + j] = dist(rng);
	    B.elems[i*A.rows + j] = dist(rng);
	}
    }

    CPUMatMul(&A, &B, &C);

    delete[] A.elems;
    delete[] B.elems;
    delete[] C.elems;
};
