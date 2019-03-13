#include <iostream>
#include <omp.h>
#include <conio.h>
#include <iomanip>
#include <time.h>
#include <ctime>
#include <random>

std::default_random_engine generator(time(0));
std::uniform_real_distribution <double> dist(0, 1);
using namespace std;

void CreateMatrix(double* matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for(int j = i; j < size; j++)
		{
			if(i == j)
				matrix[i * size + j] = dist(generator);
			else
				matrix[j * size + i] = matrix[i * size + j] =  dist(generator);
		}
	}
}
void CreateVector(double* vector, int size)
{
	for (int i = 0; i < size; i++)
	{
		vector[i] = dist(generator);
	}
}
void PrintVector(double* vector, int count)
{
	if(count <= 20)
	{
		for (int i = 0; i < count; i++)
		{
			std::cout << std::setprecision(3) << vector[i] << "\t";
		}
		std::cout << std::endl;
	}
	else
		std::cout << "Vector size is too large" << std::endl;
}
void Print_SLU(double* matrix, double* vector, int Size)
{
	if(Size <= 20)
	{
		for (int i = 0; i < Size; i++)
		{
			for (int j = 0; j < Size; j++)
			{
				std::cout << std::setprecision(3) << matrix[Size * i + j] << "\t";
			}
			//std::cout << "\t";
			std::cout << " | " << std::setprecision(3) << vector[i] << "\n";
		}
	}
	else
		std::cout << "System size is too large" << endl;
}
// Sequential functions:
void MultiplicateMV(double* matrix, double* vector, double* result, int size)
{
	for (int i = 0; i < size; i++)
	{
		result[i] = 0;
		for (int j = 0; j < size; j++)
		{
			result[i] += matrix[i * size + j] * vector[j];
		}
	}
}
double MultiplicateVV(double* vector1, double* vector2, int size)
{
	double result = 0;
	for(int i = 0; i < size; i++)
		result += vector1[i] * vector2[i];
	return result;
}
void SGMethod(double* matrix, double* vector, double* x0, double eps, double* result, int* count, int maxIter, int size)
{
	// rPrev - невязка текущего приближения
	double* rPrev = new double[size];
	// rNext - невязка следующего приближения
	double* rNext = new double[size];
	// р - вектор направления
	double* p = new double[size];
	// Вспомогательные переменные:
	// y - векторное произведение matrix * х0
	double* y = new double[size];
	// Ар - векторное произведение matrix * р
	double* Ap = new double[size];
	// check - текущая точность метода
	// norm - норма вектора правой части
	// beta, alpha - коэффициенты расчетных формул
	double beta, alpha, check, norm;
	double* swap;
	// Инициализация метода
	*count = 0;
	norm = sqrt(MultiplicateVV(vector, vector, size));
	// вычисление невязки с начальным приближением
	MultiplicateMV(matrix, x0, y, size);
	for(int i = 0; i < size; i++)
		rPrev[i] = vector[i] - y[i];
	memcpy(p, rPrev, size * sizeof(double));
	memcpy(result, x0, size * sizeof(double));
	// Выполнение итераций метода
	do
	{
		(*count)++;
		MultiplicateMV(matrix, p, Ap, size);
		alpha = MultiplicateVV(rPrev, rPrev, size) / MultiplicateVV(p, Ap, size);
		for (int i = 0; i < size; i++)
		{
			result[i] += alpha * p[i];
			rNext[i] = rPrev[i] - alpha * Ap[i];
		}
		beta = MultiplicateVV(rNext, rNext, size) / MultiplicateVV(rPrev, rPrev, size);
		check = sqrt(MultiplicateVV(rNext, rNext, size));
		for (int j = 0; j < size; j++)
			p[j] = beta * p[j] + rNext[j];
		swap = rNext;
		rNext = rPrev;
		rPrev = swap;
	}
	while ((check > eps) && (*(count) <= maxIter));

	delete[] rPrev;
	delete[] rNext;
	delete[] p;
	delete[] y;
	delete[] Ap;
}
// OMP functions:
double MultiplicateVV_OMP(double* a, double* b, int size) 
{ 
	double result = 0; 
	#pragma omp parallel for reduction(+:result) 
	for (int i = 0; i < size; i++) 
	{ 
		result += a[i] * b[i]; 
	} 
	return result; 
}
void MultiplicateMV_OMP(double* matrix, double* vector, double* result, int size)
{
	int i, j;
	#pragma omp parallel for private(i, j)
	for (i = 0; i < size; i++) 
	{
		result[i] = 0;
		for (j = 0; j < size; j++)
		{
			result[i] += matrix[i * size + j] * vector[j];
		}
	}
}
void SGMethod_OMP(double* matrix, double* vector, double* x0, double eps, double* result, int* count, int maxIter, int size)
{
	// rPrev - невязка текущего приближения
	double* rPrev = new double[size];
	// rNext - невязка следующего приближения
	double* rNext = new double[size];
	// р - вектор направления
	double* p = new double[size];
	// Вспомогательные переменные:
	// y - векторное произведение matrix * х0
	double* y = new double[size];
	// Ар - векторное произведение matrix * р
	double* Ap = new double[size];
	// check - текущая точность метода
	// norm - норма вектора правой части
	// beta, alpha - коэффициенты расчетных формул
	double beta, alpha, check, norm;
	double* swap;
	// Инициализация метода
	*count = 0;
	norm = sqrt(MultiplicateVV_OMP(vector, vector, size));
	// вычисление невязки с начальным приближением
	MultiplicateMV_OMP(matrix, x0, y, size);
	for(int i = 0; i < size; i++)
		rPrev[i] = vector[i] - y[i];
	memcpy(p, rPrev, size * sizeof(double));
	memcpy(result, x0, size * sizeof(double));
	// Выполнение итераций метода
	do
	{
		(*count)++;
		MultiplicateMV_OMP(matrix, p, Ap, size);
		alpha = MultiplicateVV_OMP(rPrev, rPrev, size) / MultiplicateVV_OMP(p, Ap, size);
		for (int i = 0; i < size; i++)
		{
			result[i] += alpha * p[i];
			rNext[i] = rPrev[i] - alpha * Ap[i];
		}
		beta = MultiplicateVV_OMP(rNext, rNext, size) / MultiplicateVV_OMP(rPrev, rPrev, size);
		check = sqrt(MultiplicateVV_OMP(rNext, rNext, size));
		for (int j = 0; j < size; j++)
			p[j] = beta * p[j] + rNext[j];
		swap = rNext;
		rNext = rPrev;
		rPrev = swap;
	}
	while ((check > eps) && (*(count) <= maxIter));

	delete[] rPrev;
	delete[] rNext;
	delete[] p;
	delete[] y;
	delete[] Ap;
}

void main(int argc, char**argv)
{
	int treadsNum;
	int size;
	double eps;
	int count_seq, count_omp;
	int maxIter_seq = 0, maxIter_omp = 0;
	clock_t start_seq, finish_seq;
	double time_seq;
	double start_omp, finish_omp, time_omp;

	cout << "Enter number of treads: ";
	cin >> treadsNum;
	cout << "Enter size: ";
	cin >> size;
	cout << "Enter eps: ";
	cin >> eps;

	double* matrix = new double[size * size];
	double* vector = new double[size];
	double* result_omp = new double[size];
	double* result_seq = new double[size];
	double* x0_seq = new double[size];
	double* x0_omp = new double[size];

	for(int i = 0; i < size; i++)
	{
		x0_seq[i] = 0;
		x0_omp[i] = 0;
	}
	if(maxIter_seq == 0)
		maxIter_seq = size * 10;
	if(maxIter_omp == 0)
		maxIter_omp = size * 10;

	CreateMatrix(matrix,size);
	CreateVector(vector,size);
	cout << "SYSTEM: " << endl;
	Print_SLU(matrix,vector,size);

	// Последовательная версия
	start_seq = clock();
	SGMethod(matrix,vector,x0_seq,eps,result_seq,&count_seq,maxIter_seq,size);
	finish_seq = clock();
	time_seq = (double)(finish_seq - start_seq)/CLOCKS_PER_SEC;

	// Параллельная версия
	omp_set_num_threads(treadsNum);

	start_omp = omp_get_wtime();
	SGMethod_OMP(matrix,vector,x0_omp,eps,result_omp,&count_omp,maxIter_omp,size);
	finish_omp = omp_get_wtime();
	time_omp = finish_omp - start_omp;

	#pragma omp critical

	cout << "SEQUENTIAL ALGORITHM: " << endl;
	cout << "Result: " << endl;
	PrintVector(result_seq,size);
	cout << "Count: " << count_seq << endl;
	cout << "Time: " << time_seq << " c" << endl;
	cout << "PARALLEL ALGORITHM: " << endl;
	cout << "Result: " << endl;
	PrintVector(result_omp,size);
	cout << "Count: " << count_omp << endl;
	cout << "Time: " << time_omp << " c" << endl;
	cout << "Acceleration: " << time_seq / time_omp << endl;

	// Проверка результатов
	bool corr = false;
	int j = 0;
	int max = 0;
	double err;
	for(int i = 0; i < size; i++)
	{
		err = abs(result_omp[i] - result_seq[i]);
		if(err <= eps)
			corr = true;
		else
		{
			corr = false;
			j++;
		}
		if(err > max)
			max = err;
	}
	if(corr && (j==0))
	{
		std::cout << "Calculations are correct" << std::endl;
		std::cout << "Error: " << err << std::endl;
	}
	else
	{
		std::cout << "Calculations aren't correct!" << std::endl;
		std::cout << "Error: " << err << std::endl;
	}

	delete[] matrix;
	delete[] vector;
	delete[] result_omp;
	delete[] result_seq;
	delete[] x0_omp;
	delete[] x0_seq;
}
