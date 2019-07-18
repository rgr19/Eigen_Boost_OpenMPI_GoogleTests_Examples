//
// Created by egrzrbr on 2019-04-26.
//

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
/*****************
 * Main function *
 *****************/

#include <unistd.h>
#include <time.h>
#include <iostream>

#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <Eigen/Dense>

using namespace Eigen;

/**********************************
 * Pseudo-random number generator *
 **********************************/

#define EIGEN_COUT(mat) {std::cout << #mat << " = {\n" << mat << "}" << std::endl;}

static uint64_t mat_rng[2] = {11ULL, 1181783497276652981ULL};

static inline uint64_t xorshift128plus(uint64_t s[2]) {

	uint64_t x, y;
	x = s[0], y = s[1];
	s[0] = y;
	x ^= x << 23;
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	y += s[1];
	return y;
}

double mat_drand(void) {

	return (xorshift128plus(mat_rng) >> 11) * (1.0 / 9007199254740992.0);
}

template<class Matrix_t>
void mat_gen_random_ublas(Matrix_t &m) {

	ssize_t i, j;
	for (i = 0; i < m.rows(); ++i)
		for (j = 0; j < m.cols(); ++j)
			m(i, j) = mat_drand();
}

template<class Matrix_t>
void mat_arange(Matrix_t &m) {

	const ssize_t N = m.rows();
	const ssize_t M = m.cols();
	ssize_t i, j;
	for (i = 0; i < N; ++i)
		for (j = 0; j < M; ++j)
			m(i, j) = j * N + i;
}

template<typename T, size_t N, size_t TRIALS>
void test_trials(const char opt = 'r') {


	int c;
	clock_t t;

	{

		MatrixXf a(N, N), b(N, N), m(N, N);

		if (opt == 'r') {
			mat_gen_random_ublas(a);
			mat_gen_random_ublas(b);
		} else {
			mat_arange(a);
			mat_arange(b);
		}

		t = clock();
		for (int i = 0; i < TRIALS; i++)
			m = a * b;
		fprintf(stderr, "CPU time: %g\n", (double) (clock() - t) / CLOCKS_PER_SEC);

		if (N > 10) return;
		EIGEN_COUT(a)
		EIGEN_COUT(b)
		EIGEN_COUT(c)
	}

	{

		Matrix<float, N, N> a, b, c;

		if (opt == 'r') {
			mat_gen_random_ublas(a);
			mat_gen_random_ublas(b);
		} else {
			mat_arange(a);
			mat_arange(b);
		}


		t = clock();

		for (int i = 0; i < TRIALS; i++)
			c = a * b;
		fprintf(stderr, "CPU time: %g\n", (double) (clock() - t) / CLOCKS_PER_SEC);
		if (N > 10) return;
		EIGEN_COUT(a)
		EIGEN_COUT(b)
		EIGEN_COUT(c)
	}

}

template<typename T, size_t N>
void matNN_mul_matNN(const char opt = 'r') {

	std::cout << "matNN_mul_matNN" << std::endl;

	clock_t t;
	Matrix<T, N, N> a;
	Matrix<T, N, N> b;
	Matrix<T, N, N> c;

	if (opt == 'r') {
		mat_gen_random_ublas(a);
		mat_gen_random_ublas(b);
	} else {
		mat_arange(a);
		mat_arange(b);
	}

	t = clock();

	c = a * b;

	std::cout << c.size() << std::endl;
	fprintf(stderr, "CPU time: %g\n", (double) (clock() - t) / CLOCKS_PER_SEC);
	if (N > 10) return;
	EIGEN_COUT(a)
	EIGEN_COUT(b)
	EIGEN_COUT(c)
}

template<typename T, size_t N>
void vecN1_mul_vec1N(const char opt = 'r') {

	clock_t t;
	Matrix<T, N, 1> a;
	Matrix<T, 1, N> b;
	Matrix<T, N, N> c;

	if (opt == 'r') {
		mat_gen_random_ublas(a);
		mat_gen_random_ublas(b);
	} else {
		mat_arange(a);
		mat_arange(b);
	}
	t = clock();

	c = a * b;

	std::cout << c.size() << std::endl;
	fprintf(stderr, "CPU time: %g\n", (double) (clock() - t) / CLOCKS_PER_SEC);
	if (N > 10) return;
	EIGEN_COUT(a)
	EIGEN_COUT(b)
	EIGEN_COUT(c)
}


template<typename T, size_t N, size_t M>
void matNM_mul_vecM1(const char opt = 'r') {

	clock_t t;
	Matrix<T, N, M> a;
	Matrix<T, M, 1> b;
	Matrix<T, N, 1> c;

	if (opt == 'r') {
		mat_gen_random_ublas(a);
		mat_gen_random_ublas(b);
	} else {
		mat_arange(a);
		mat_arange(b);
	}

	t = clock();

	c = a * b;

	std::cout << c.size() << std::endl;
	fprintf(stderr, "CPU time: %g\n", (double) (clock() - t) / CLOCKS_PER_SEC);
	if (N > 10) return;
	EIGEN_COUT(a)
	EIGEN_COUT(b)
	EIGEN_COUT(c)
}


template<typename T, size_t N>
void vec1N_mul_vecN1(const char opt = 'r') {

	clock_t t;
	Matrix<T, 1, N> a;
	Matrix<T, N, 1> b;
	Matrix<T, 1, 1> c;

	if (opt == 'r') {
		mat_gen_random_ublas(a);
		mat_gen_random_ublas(b);
	} else {
		mat_arange(a);
		mat_arange(b);
	}

	t = clock();

	c = a * b;

	std::cout << c.size() << std::endl;
	fprintf(stderr, "CPU time: %g\n", (double) (clock() - t) / CLOCKS_PER_SEC);
	if (N > 10) return;
	EIGEN_COUT(a)
	EIGEN_COUT(b)
	EIGEN_COUT(c)
}


int main(int argc, char *argv[]) {

	constexpr int N = 500;
	constexpr int M = 400;
	constexpr int K = 10;
	clock_t t;

	matNN_mul_matNN<float, N>();
	matNM_mul_vecM1<float, N, M>();
	vecN1_mul_vec1N<float, N>();
	vec1N_mul_vecN1<float, N>();


	matNN_mul_matNN<float, 5>(0);
	matNM_mul_vecM1<float, 5, 4>(0);
	vecN1_mul_vec1N<float, 5>(0);
	vec1N_mul_vecN1<float, 5>(0);

	return 0;
}
