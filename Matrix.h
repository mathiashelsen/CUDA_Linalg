#ifndef _MATRIX_H
#define _MATRIX_H

#include <cuda.h>
#include <cuda_runtime.h>

enum MemoryLocation
{
    MemoryLocationCPU, 
    MemoryLocationGPU
};

class Matrix
{
public:
    int rows;
    int cols;
    float *elems;
    MemoryLocation loc;

    Matrix(int _rows, int _cols, MemoryLocation _loc);
    ~Matrix();

    void free();
};

#endif 
