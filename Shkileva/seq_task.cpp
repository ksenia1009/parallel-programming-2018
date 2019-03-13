#include "stdafx.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <ctime>

std::default_random_engine generator(time(0));
std::uniform_real_distribution <double> dist(0, 10);
void CreateMatrix(double** matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for(int j = i; j < size; j++)
		{
			if(i == j)
				matrix[i][j] = dist(generator);
			else
				matrix[j][i] = matrix[i][j] = dist(generator);
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
void PrintVector(double* vector, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << std::setprecision(3) << vector[i] << "\t";
	}
	std::cout << std::endl;	
}
void Print_SLU(double** matrix, double* vector, int size)
{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				std::cout << std::setprecision(3) << matrix[i][j] << "\t";
			}
			//std::cout << "\t";
			std::cout << " | " << std::setprecision(3) << vector[i] << "\n";
		}	
}
void MultiplicateMV(double** matrix, double* vector, double* result, int size)
{
	for (int i = 0; i < size; i++) 
	{
		result[i] = 0;
		for (int j = 0; j < size; j++)
			result[i] += matrix[i][j] * vector[j];
	}
}
double MultiplicateVV(double* vector1, double* vector2, int size)
{
	double result = 0;
	for(int i = 0; i < size; i++)
		result += vector1[i] * vector2[i];
	return result;
}
double Norm(double** matrix, double* res, double* vector, int size)
{
	double norm = 0;
	double r = 0; 
	double* err = new double[size];
	MultiplicateMV(matrix, res, err, size); // matrix * res
	for (int i = 0; i < size; i++)
		err[i] = err[i] - vector[i]; // matrix * res - b
	r = MultiplicateVV(err, err, size);
	norm = sqrt(r);
	delete[] err;
	return norm;
}
void SGMethod(double** matrix, double* vector, double* x0, double eps, double* result, int* count, int maxIter, int size)
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
	double* res = new double[size];
	double beta, alpha, check, norm;
	double* swap;
	// Инициализация метода
	*count = 0;
	norm = sqrt(MultiplicateVV(vector, vector, size));
	// Вычисление невязки с начальным приближением
	MultiplicateMV(matrix, x0, y, size);
	for(int i = 0; i < size; i++)
		rPrev[i] = vector[i] - y[i];
	memcpy(p, rPrev, size * sizeof(double));
	memcpy(result, x0, size * sizeof(double));
	int max;
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
		// Вычисление нормы невязки
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
int _tmain(int argc, _TCHAR* argv[])
{
	double** matrix; // исходная матрица
	double* vector; // вектор правой части
	double* result; // вектор-результат
	double* x0; // начальное приближение
	double eps; // точность
	int size; // размер матрицы
	int maxIter = 0; // максимальное количество итераций
	int count; // количество итераций
	clock_t start, finish;
	double time;
	double* res;

	std::cout << "Enter size: ";
	std::cin >> size;
	std::cout << "Enter eps: ";
	std::cin >> eps;

	matrix = new double*[size];
	for(int i = 0; i < size; i++)
		matrix[i] = new double[size];
	vector = new double[size];
	result = new double[size];
	x0 = new double[size];
	res = new double[size];
	for(int i = 0; i < size; i++)
		x0[i] = 0;
	if(maxIter == 0)
		maxIter = size * 10;

	// вывод системы 
	CreateMatrix(matrix, size);
	CreateVector(vector, size);
	std::cout << "System of linear algebraic equations:" << std::endl;
	if(size < 15)
		Print_SLU(matrix, vector, size);
	else
		std::cout << "System size is too large" << std::endl;

	// решение СЛАУ методом сопряженных градиентов
	start = clock();
	SGMethod(matrix, vector, x0, eps, result, &count, maxIter, size);
	finish = clock();
	time = (double)(finish - start)/CLOCKS_PER_SEC;

	// вывод решения
	std::cout << "Solution:" << std::endl;
	if(size < 15)
		PrintVector(result, size);
	std::cout << "Time: " << time << " c" << std::endl;
	std::cout << "Iterations count: " << count << std::endl;

	// проверка результов
	MultiplicateMV(matrix,result,res,size);
	bool corr = false;
	int j = 0;
	double err;
	int max = 0;
	for(int i = 0; i < size; i++)
	{
		err = abs(vector[i] - res[i]);
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
	delete[] result;
	delete[] x0;
	return 0;
}
