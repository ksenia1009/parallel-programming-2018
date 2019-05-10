#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <iostream>
#include <time.h>
#include <ctime>
#include <random>

std::default_random_engine generator(time(0));
std::uniform_real_distribution <double> dist(1,5);

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
void PrintMatrix(double* matrix, int size)
{
	if(size <= 20)
	{
		for (int i = 0; i < size; i++) 
		{
			for (int j = 0; j < size; j++) 
			{
				std::cout << matrix[size * i + j] << "\t";
			}
			std::cout << std::endl;
		}
	}
	else
		std::cout << "Matrix size is too large" << std::endl;
}
void PrintVector(double* vector, int count)
{
	if(count <= 20)
	{
		for (int i = 0; i < count; i++)
		{
			std::cout << vector[i] << "\t";
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
				std::cout << matrix[Size * i + j] << "\t";
			}
			std::cout << " | " << vector[i] << "\n";
		}
	}
	else
		std::cout << "System size is too large" << std::endl;
}
// TBB Functions
double VectorsMultiplication(const double *v1, const double *v2, int size)
{
	double result = 0.0;
	for(int i = 0; i < size; i++)
		result += v1[i] * v2[i];
	return result;
}
class VectorsMultiplicator
{
	double *matrix, *vector;
	double *resultVector;
	int size;
public:
	VectorsMultiplicator(double *tmatrix, double *tvector, double *tresultVector, int tsize) : matrix(tmatrix), vector(tvector), resultVector(tresultVector), size(tsize) {}
	void operator()(const tbb::blocked_range<int>& r) const
	{
		int begin = r.begin(), end = r.end();
		for(int i = begin; i < end; i++)
		{
			resultVector[i] = VectorsMultiplication(&(matrix[i*size]), vector, size);
		}
	}
};
class ScalarMultiplicator 
{
private:
	const double *a, *b;
	double c;
public:
	explicit ScalarMultiplicator(double *ta, double *tb) : a(ta), b(tb), c(0) {}
	ScalarMultiplicator(const ScalarMultiplicator& m, tbb::split) : a(m.a), b(m.b), c(0) {}
	void operator()(const tbb::blocked_range<int>& r)
	{
		int begin = r.begin(), end = r.end();
		c += VectorsMultiplication(&(a[begin]), &(b[begin]), end - begin);
	}
	void join(const ScalarMultiplicator& multiplicator)
	{
		c += multiplicator.c;
	}
	double Result()
	{
		return c;
	}
};
void TBBMatrixVectorMultiplication(double* matrix, double* vector, double* resultVector, int size, int grainSize)
{
	tbb::parallel_for(tbb::blocked_range<int>(0, size, grainSize), VectorsMultiplicator(matrix, vector, resultVector, size));
}
double TBBScalarMultiplication(double* vector1, double* vector2, int size, int grainSize)
{
	ScalarMultiplicator s(vector1, vector2);
	tbb::parallel_reduce(tbb::blocked_range<int>(0, size, grainSize), s);
	return s.Result();
}
void SGMethod_TBB(double* matrix, double* vector, double* x0, double eps, double* result, int* count, int maxIter, int size, int grainSize)
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
	double beta, alpha, check;
	double* swap;
	// Инициализация метода
	*count = 0;
	for(int i = 0; i < size; i++)
	{
		rPrev[i] = vector[i];
		p[i] = rPrev[i];
		result[i] = 0.0;
  }
	// Выполнение итераций метода
	do
	{
		(*count)++;
		TBBMatrixVectorMultiplication(matrix, p, Ap, size, grainSize);
		alpha = TBBScalarMultiplication(rPrev, rPrev, size, grainSize) / TBBScalarMultiplication(p, Ap, size, grainSize);
		for (int i = 0; i < size; i++)
		{
			result[i] += alpha * p[i];
			rNext[i] = rPrev[i] - alpha * Ap[i];
		}
		beta = TBBScalarMultiplication(rNext, rNext, size, grainSize) / TBBScalarMultiplication(rPrev, rPrev, size, grainSize);
		check = sqrt(TBBScalarMultiplication(rNext, rNext, size, grainSize));
		for (int j = 0; j < size; j++)
			p[j] = beta * p[j] + rNext[j];
		swap = rNext;
		rNext = rPrev;
		rPrev = swap;
	}
	while ((check > eps) && (*count < maxIter));

	delete[] rPrev;
	delete[] rNext;
	delete[] p;
	delete[] y;
	delete[] Ap;
}
// Sequential Functions
void MultiplicateMV(const double* matrix, const double* vector, double* result, int size)
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
double MultiplicateVV(const double* vector1, const double* vector2, int size)
{
	double result = 0;
	for(int i = 0; i < size; i++)
	{
		result += vector1[i] * vector2[i];
	}
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
	double beta, alpha, check;
	double* swap;
	// Инициализация метода
	*count = 0;
	for(int i = 0; i < size; i++)
	{
		rPrev[i] = vector[i];
		p[i] = rPrev[i];
		result[i] = 0.0;
	}
	// Выполнение итераций метода
	do
	{
		(*count)++;
		MultiplicateMV(matrix, p, Ap, size);
		alpha = VectorsMultiplication(rPrev, rPrev, size) / VectorsMultiplication(p, Ap, size);
		for (int i = 0; i < size; i++)
		{
			result[i] += alpha * p[i];
			rNext[i] = rPrev[i] - alpha * Ap[i];
		}
		beta = VectorsMultiplication(rNext, rNext, size) / VectorsMultiplication(rPrev, rPrev, size);
		check = sqrt(VectorsMultiplication(rNext, rNext, size));
		for (int j = 0; j < size; j++)
			p[j] = beta * p[j] + rNext[j];
		swap = rNext;
		rNext = rPrev;
		rPrev = swap;
	}
	while ((check > eps) && (*count < maxIter));

	delete[] rPrev;
	delete[] rNext;
	delete[] p;
	delete[] y;
	delete[] Ap;
}
int main()
{
	int threadNum;
	int size;
	int grainSize;
	double eps;
	double *vector, *result_seq, *result_tbb, *x0_seq, *x0_tbb;
	double *matrix;
	int count_tbb = 0, count_seq = 0;
	int maxIter_tbb = 0, maxIter_seq = 0;
	clock_t start_seq, finish_seq, start_tbb, finish_tbb;
	double time_seq, time_tbb;

	std::cout << "Enter a number of threads: ";
	std::cin >> threadNum;
	std::cout << "Enter size: ";
	std::cin >> size;
	std::cout << "Enter eps: ";
	std::cin >> eps;

	tbb::task_scheduler_init init(threadNum);

	vector = new double[size];
	result_seq = new double[size];
	result_tbb = new double[size];
	x0_seq = new double[size];
	x0_tbb = new double[size];
	matrix = new double[size*size];
	
	// Размер порции вычислений 
	grainSize = 5;

	maxIter_seq = size * 10;
	maxIter_tbb = size * 10;

	for(int i = 0; i < size; i++)
	{
		x0_seq[i] = 0.0;
		x0_tbb[i] = 0.0;
	}

	CreateMatrix(matrix,size);
	CreateVector(vector,size);
	std::cout << "SYSTEM: " << std::endl;
	Print_SLU(matrix, vector, size);

	// Последовательная версия
	start_seq = clock();
	SGMethod(matrix, vector, x0_seq, eps, result_seq, &count_seq, maxIter_seq, size);
	finish_seq = clock();
	time_seq = (double)(finish_seq - start_seq)/CLOCKS_PER_SEC;
	// Параллельная версия
	start_tbb = clock();
	SGMethod_TBB(matrix, vector, x0_tbb, eps, result_tbb, &count_tbb, maxIter_tbb, size, grainSize);
	finish_tbb = clock();
	time_tbb = (double)(finish_tbb - start_tbb)/CLOCKS_PER_SEC;

	// Вывод результатов
	std::cout << "SEQUENTIAL ALGORITHM: " << std::endl;
	std::cout << "Result: " << std::endl;
	PrintVector(result_seq, size);
	std::cout << "Count: " << count_seq << std::endl;
	std::cout << "Time: " << time_seq << " c" << std::endl;
	std::cout << "PARALLEL ALGORITHM: " << std::endl;
	std::cout << "Result: " << std::endl;
	PrintVector(result_tbb, size);
	std::cout << "Count: " << count_tbb << std::endl;
	std::cout << "Time: " << time_tbb << " c" << std::endl;
	std::cout << "Acceleration: " << time_seq / time_tbb << std::endl;

	// Проверка результатов
	bool corr = false;
	int j = 0;
	int max = 0;
	double err = 0;
	for(int i = 0; i < size; i++)
	{
		err = abs(result_tbb[i] - result_seq[i]);
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
