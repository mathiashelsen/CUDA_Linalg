#ifndef _MATRIX_H
#define _MATRIX_H

typedef struct
{
    int cols;
    int rows;
    float *elems;
} Matrix;

/*template<T> class Matrix
{
    int rows;
    int cols;
    T	*elems;

    public:
	Matrix(int _rows, int _cols);
	~Matrix();
	void operator+=(S B);
	void operator*=(S B);
};
*/
//void InvokeMatMul(Matrix *A, Matrix *B, Matrix *C);

#endif 
