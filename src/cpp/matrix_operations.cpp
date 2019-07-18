// Example program
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <bits/uniform_int_dist.h>
#include <random>
#include <algorithm>

#include "MUtil.h"

static std::default_random_engine randEng;

using DIST_UI = std::uniform_int_distribution<int>;
using DIST_UR = std::uniform_real_distribution<float>;
using DIST_RN = std::normal_distribution<float>;

inline const auto rand_uniform_int(const int low, const int high) {

	static DIST_UI distUI{};
	return distUI(randEng, DIST_UI::param_type{low, high});
}

inline const auto rand_uniform_real(const float low, const float high) {

	static DIST_UR distUR{};
	return distUR(randEng, DIST_UR::param_type{low, high});
}

inline const auto rand_normal(const float mean, const float stdev) {

	static DIST_RN distRN{};
	return distRN(randEng, DIST_RN::param_type{mean, stdev});
}

int test_small() {

	constexpr size_t shapeNN[3] = {4, 3, 2};

	constexpr size_t TRIALS = (int) 1e6;
	constexpr size_t N = 4;
	constexpr size_t M = 3 ;
	constexpr size_t K = 2  ;
	constexpr size_t mB = 2  ;
	#define DEBUG 0

	float W_1[M][N] = {{0.88202617,  0.2000786,   0.48936899, 1.1204466},
	                   {0.933779,    -0.48863894, 0.47504421, -0.0756786},
	                   {-0.05160943, 0.20529925,  0.07202179, 0.72713675}};
	float V_0[N][mB] = {{0.3130677,   -0.85409574},
	                    {-2.55298982, 0.6536186},
	                    {0.8644362,   -0.74216502},
	                    {2.26975462,  -1.45436567}};
	float b_1[M][mB] = {{2.73150239, -2.61529132},
	                    {1.77869983, -1.35941699},
	                    {1.17239611, -0.93270799}};
	float W_2[K][M] = {{0.43938534, 0.0702491,  0.25626456},
	                   {0.19264696, 0.86260696, -0.11844818}};
	float V_1[M][mB] = {{2.73150239, -2.61529132},
	                    {1.77869983, -1.35941699},
	                    {1.17239611, -0.93270799}};
	float b_2[K][mB] = {{1.62557773, -1.48363848},
	                    {1.9216663,  -1.56599292}};

	auto W_1_ = std::array<float, M * N>();
	auto b_1_ = std::array<float, M * mB>();

	auto V_0_ = std::array<float, N * mB>();

	for (int j = 0; j < M; j++) {
		for (int i = 0; i < N; i++) {
			W_1_[j * N + i] = W_1[j][i];
		}
	}

	for (int k = 0; k < mB; k++) {
		for (int i = 0; i < N; i++) {
			if constexpr (DEBUG > 0) std::cout << "...Vp[" << k * N + i << "] = V[" << i << "][" << k << "]" << std::endl;
			V_0_[k * N + i] = V_0[i][k];
		}
	}
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
//				std::cout << "...b[" << k * M + j << "] += W[" << j * N + i << "] * V[" << k * N + i << "]" << std::endl;
				b_1_[k * M + j] += W_1_[j * N + i] * V_0_[k * N + i];
			}
			if constexpr (DEBUG > 0) std::cout << "b[" << j * mB + k << "] = " << b_1_[j * mB + k] << " : " << b_1[j][k] << std::endl;
		}
	}

	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_A[" << k * M + j << "] = " << b_1_[k * M + j] << std::endl;

		}
	}

	TIME_START

	for (int d = 0; d < (int) TRIALS; d++) {
		auto f = rand_uniform_real(-1, 1);
		for (int k = 0; k < mB; k++) {
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < N; i++) {
					if constexpr (DEBUG > 0) std::cout << "...b[" << j * mB + k << "] += W[" << j * N + i << "] * V[" << k * N + i << "]" << std::endl;
					b_1_[k * M + j] += W_1_[j * N + i] * V_0_[k * N + i] - f;
				}
				if constexpr (DEBUG > 0) std::cout << "b[" << j * mB + k << "] = " << b_1_[j * mB + k] << " : " << b_1[j][k] << std::endl;
			}
		}
	}
	std::cout << "A : ";
	TIME_CHECK


	b_1_.fill(0);

	auto W_1_begin = W_1_.begin();
	auto V_0_begin = V_0_.begin();
	auto b_1_begin = b_1_.begin();
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
//				std::cout << "...b[" << k * M + j << "] += W[" << j * N + i << "] * V[" << k * N + i << "]" << std::endl;
				b_1_begin[k * M + j] += W_1_begin[j * N + i] * V_0_begin[k * N + i];
			}
			if constexpr (DEBUG > 0) std::cout << "b[" << j * mB + k << "] = " << b_1_[j * mB + k] << " : " << b_1[j][k] << std::endl;
		}
	}

	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[k * M + j] << std::endl;

		}
	}

	TIME_START

	for (int d = 0; d < (int) TRIALS; d++) {
		auto f = rand_uniform_real(-1, 1);
		for (int k = 0; k < mB; k++) {
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < N; i++) {
					if constexpr (DEBUG > 0) std::cout << "...b[" << j * mB + k << "] += W[" << j * N + i << "] * V[" << k * N + i << "]" << std::endl;
					b_1_[j * mB + k] += W_1_begin[j * N + i] * V_0_begin[k * N + i] - f;
				}
				if constexpr (DEBUG > 0) std::cout << "b[" << j * mB + k << "] = " << b_1_[j * mB + k] << " : " << b_1[j][k] << std::endl;
			}
		}
	}


	std::cout << "B : ";
	TIME_CHECK


	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();
	auto b_1_id = 0;
	auto V_0_id = 0;
	auto W_1_id = 0;
	for (int k = 0; k < mB; k++) {
		V_0_id = 0;
		W_1_id = 0;
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
//				std::cout << "...b[" << b_1_id << "] += W[" << W_1_id << "] * V[" << V_0_id << "]" << std::endl;
				b_1_begin[b_1_id] += W_1_begin[W_1_id++] * V_0_begin[V_0_id++];
			}
//			std::advance(V_0_begin, -N);
			//std::advance(b_1_begin, 1);
			b_1_id++;
			V_0_id -= N;
		}
//		std::advance(V_0_begin, M);
//		std::advance(W_1_begin, -N*M);
		V_0_id += N;

	}

	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	TIME_START
	for (int d = 0; d < (int) TRIALS; d++) {
		V_0_id = 0;
		b_1_id = 0;
		auto f = rand_uniform_real(-1, 1);
		for (int k = 0; k < mB; k++) {
			W_1_id = 0;
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < N; i++) {
					b_1_begin[b_1_id] += W_1_begin[W_1_id++] * V_0_begin[V_0_id++] - f;
				}
				b_1_id++;
				V_0_id -= N;
			}
			V_0_id += N;

		}

	}

	std::cout << "C : ";
	TIME_CHECK

	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();
	b_1_id = 0;
	V_0_id = 0;
	W_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			float sum = 0;
			for (int i = 0; i < N; i++) {
//				std::cout << "...b[" << b_1_id << "] += W[" << W_1_id++ << "] * V[" << V_0_id++ << "]" << std::endl;
				const float w1 = *W_1_begin;
				const float v0 = *V_0_begin;
				sum += w1 * v0;
				W_1_begin++;
				V_0_begin++;
			}
			*b_1_begin++ = sum;
			V_0_begin -= N;

			b_1_id++;
			V_0_id -= N;
		}
		V_0_begin += N;
		W_1_begin -= N * M;
		V_0_id += N;
		W_1_id -= N * M;

	}

	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();
	auto V_0_begin_ = V_0_.begin();
	auto b_1_begin_ = b_1_.begin();
	auto W_1_begin_ = W_1_.begin();
	TIME_START

	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;
		auto f = rand_uniform_real(-1, 1);
		for (int k = 0; k < mB; k++) {
			W_1_begin = W_1_begin_;
			for (int j = 0; j < M; j++) {
				float sum = 0;
				for (int i = 0; i < N; i++) {
					const float w1 = *W_1_begin++;
					const float v0 = *V_0_begin++;
					sum += w1 * v0 - f;
				}
				*b_1_begin++ = sum;
				V_0_begin -= N;
			}
			V_0_begin += N;
		}
	}
	std::cout << "D : ";

	TIME_CHECK

	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	auto V_0_end = V_0_.end();
	auto W_1_end = W_1_.end();
	auto b_1_end = b_1_.end();

	using W_1_it = decltype(W_1_begin);
	using V_0_it = decltype(V_0_begin);
	using b_1_it = decltype(b_1_begin);

	b_1_.fill(0);

	b_1_id = 0;
	V_0_id = 0;
	W_1_id = 0;
	for (int k = 0; k < mB; k++) {

		W_1_it w1It = W_1_begin;
		for (b_1_it b1It = b_1_begin; b1It != b_1_begin + M; ++b1It) {
			float sum = 0;
			for (V_0_it v0It = V_0_begin; v0It != V_0_begin + N; ++v0It) {
//				std::cout << "...b[" << b_1_id << "] += W[" << W_1_id++ << "] * V[" << V_0_id++ << "]" << std::endl;
				const float w1 = *w1It++;
				const float v0 = *v0It;
				sum += w1 * v0;
			}
			*b1It = sum;
			//V_0_begin -= N;

			b_1_id++;
			V_0_id -= N;
		}
		b_1_begin += M;
		V_0_begin += N;
		V_0_id += N;
		W_1_id -= N * M;

	}
	V_0_begin -= N * mB;
	b_1_begin -= M * mB;

	std::cout << std::distance(V_0_begin, V_0_.begin()) << std::endl;


	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);

	TIME_START


	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	for (int d = 0; d < TRIALS; d++) {
		auto f = rand_uniform_real(-1, 1);
		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			for (b_1_it b1It = b_1_begin; b1It != b_1_begin + M; ++b1It) {
				float sum = 0;
				for (V_0_it v0It = V_0_begin; v0It != V_0_begin + N; ++v0It) {
					sum += (*w1It++) * (*v0It) - f;
				}
				*b1It = sum;
				b_1_id++;
			}
			b_1_begin += M;
			V_0_begin += N;
		}
		V_0_begin -= N * mB;
		b_1_begin -= M * mB;
	}

	std::cout << "E : ";
	TIME_CHECK

	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();

	V_0_begin = V_0_begin_;
	b_1_begin = b_1_begin_;

	for (int k = 0; k < mB; k++) {
		W_1_it w1It = W_1_begin;
		for (b_1_it b1It = b_1_begin + (M * k); b1It != b_1_begin + (M * (k + 1)); ++b1It) {
			for (V_0_it v0It = V_0_begin + (N * k); v0It != V_0_begin + (N * (k + 1)); ++v0It) {
				*b1It += *w1It++ * *v0It;
			}
		}
	}


	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	TIME_START

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();

	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;
		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			for (b_1_it b1It = b_1_begin + M * k; b1It != b_1_begin + M * (k + 1); ++b1It) {
				for (V_0_it v0It = V_0_begin + N * k; v0It != V_0_begin + N * (k + 1); ++v0It) {
					*b1It += *w1It++ * *v0It;
				}
			}
		}
	}


	std::cout << "F : ";
	TIME_CHECK

	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();

	V_0_begin = V_0_begin_;
	b_1_begin = b_1_begin_;

	for (int k = 0; k < mB; k++) {
		W_1_it w1It = W_1_begin;
		for (b_1_it b1It = b_1_begin + (M * k); b1It != b_1_begin + (M * (k + 1)); ++b1It) {
			*b1It = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
			w1It += N;
		}
	}


	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	TIME_START
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();


	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;

		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			for (b_1_it b1It = b_1_begin + (M * k); b1It != b_1_begin + (M * (k + 1)); ++b1It) {
				*b1It = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
				w1It += N;
			}
		}
	}


	std::cout << "G : ";
	TIME_CHECK


	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();

	V_0_begin = V_0_begin_;
	b_1_begin = b_1_begin_;


	for (int k = 0; k < mB; k++) {
		W_1_it w1It = W_1_begin;
		std::generate(b_1_begin + (M * k), b_1_begin + (M * (k + 1)), [&V_0_begin, &w1It, k] {
			auto o = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
			w1It += N;
			return o;

		});
	}


	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	TIME_START
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();


	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;

		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			std::generate(b_1_begin + (M * k), b_1_begin + (M * (k + 1)), [&V_0_begin, &w1It, k] {
				auto o = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
				w1It += N;
				return o;

			});
		}
	}


	std::cout << "H : ";
	TIME_CHECK

	//TODO IMPROVE

	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();

	V_0_begin = V_0_begin_;
	b_1_begin = b_1_begin_;

	//IMPROVE THIS OPERATIONS
	for (int k = 0; k < mB; k++) {
		W_1_it w1It = W_1_begin;
		std::generate(b_1_begin + (M * k), b_1_begin + (M * (k + 1)), [&V_0_begin, &w1It, k] {
			auto o = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
			w1It += N;
			return o;

		});
	}


	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	TIME_START
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();

	//TODO SO THAT TIME WILL BE NICE FROM HERE

	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;

		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			std::generate(b_1_begin + (M * k), b_1_begin + (M * (k + 1)), [&V_0_begin, &w1It, k] {
				auto o = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
				w1It += N;
				return o;

			});
		}
	}





	return 0;

}


int test_big() {

	constexpr size_t shapeNN[3] = {4, 3, 2};

	constexpr size_t mB = 30;
	constexpr size_t B = 60000         ;
	constexpr size_t ITERATIONS = 30  ;
	constexpr size_t TRIALS = 100;//ITERATIONS*B/mB
	constexpr size_t N = 748            ;
	constexpr size_t M = 1000            ;
	constexpr size_t K = 10               ;

	#define DEBUG 0

	std::cout << "TRIALS = " << TRIALS << std::endl;

	auto W_1_ = std::array<float, M * N>();
	auto b_1_ = std::array<float, M * mB>();

	auto V_0_ = std::array<float, N * mB>();

	std::generate_n(W_1_.begin(), W_1_.size(), [&] { return rand_normal(0, 1); });
	std::generate_n(V_0_.begin(), V_0_.size(), [&] { return rand_normal(0, 1); });

	b_1_.fill(0);

	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
				b_1_[k * M + j] += W_1_[j * N + i] * V_0_[k * N + i];
			}
		}
	}

	TIME_START

	for (int d = 0; d < (int) TRIALS; d++) {
		auto f = rand_uniform_real(-1, 1);
		for (int k = 0; k < mB; k++) {
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < N; i++) {
					b_1_[k * M + j] += W_1_[j * N + i] * V_0_[k * N + i] - f;
				}
			}
		}
	}
	std::cout << "A : ";
	TIME_CHECK


	b_1_.fill(0);

	auto W_1_begin = W_1_.begin();
	auto V_0_begin = V_0_.begin();
	auto b_1_begin = b_1_.begin();
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
//				std::cout << "...b[" << k * M + j << "] += W[" << j * N + i << "] * V[" << k * N + i << "]" << std::endl;
				b_1_begin[k * M + j] += W_1_begin[j * N + i] * V_0_begin[k * N + i];
			}
		}
	}

	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
//			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[k * M + j] << std::endl;

		}
	}

	TIME_START

	for (int d = 0; d < (int) TRIALS; d++) {
		for (int k = 0; k < mB; k++) {
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < N; i++) {
					b_1_[j * mB + k] += W_1_begin[j * N + i] * V_0_begin[k * N + i];
				}
			}
		}
	}


	std::cout << "B : ";
	TIME_CHECK


	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();
	auto b_1_id = 0;
	auto V_0_id = 0;
	auto W_1_id = 0;
	for (int k = 0; k < mB; k++) {
		V_0_id = 0;
		W_1_id = 0;
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
//				std::cout << "...b[" << b_1_id << "] += W[" << W_1_id << "] * V[" << V_0_id << "]" << std::endl;
				b_1_begin[b_1_id] += W_1_begin[W_1_id++] * V_0_begin[V_0_id++];
			}
//			std::advance(V_0_begin, -N);
			//std::advance(b_1_begin, 1);
			b_1_id++;
			V_0_id -= N;
		}
//		std::advance(V_0_begin, M);
//		std::advance(W_1_begin, -N*M);
		V_0_id += N;

	}

	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
//			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	TIME_START
	for (int d = 0; d < (int) TRIALS; d++) {
		V_0_id = 0;
		b_1_id = 0;
		for (int k = 0; k < mB; k++) {
			W_1_id = 0;
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < N; i++) {
					b_1_begin[b_1_id] += W_1_begin[W_1_id++] * V_0_begin[V_0_id++];
				}
				b_1_id++;
				V_0_id -= N;
			}
			V_0_id += N;

		}

	}

	std::cout << "C : ";
	TIME_CHECK

	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();
	b_1_id = 0;
	V_0_id = 0;
	W_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
			float sum = 0;
			for (int i = 0; i < N; i++) {
//				std::cout << "...b[" << b_1_id << "] += W[" << W_1_id++ << "] * V[" << V_0_id++ << "]" << std::endl;
				const float w1 = *W_1_begin;
				const float v0 = *V_0_begin;
				sum += w1 * v0;
				W_1_begin++;
				V_0_begin++;
			}
			*b_1_begin++ = sum;
			V_0_begin -= N;

			b_1_id++;
			V_0_id -= N;
		}
		V_0_begin += N;
		W_1_begin -= N * M;
		V_0_id += N;
		W_1_id -= N * M;

	}

	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
//			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();
	auto V_0_begin_ = V_0_.begin();
	auto b_1_begin_ = b_1_.begin();
	auto W_1_begin_ = W_1_.begin();
	TIME_START

	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;
		for (int k = 0; k < mB; k++) {
			W_1_begin = W_1_begin_;
			for (int j = 0; j < M; j++) {
				float sum = 0;
				for (int i = 0; i < N; i++) {
					const float w1 = *W_1_begin++;
					const float v0 = *V_0_begin++;
					sum += w1 * v0;
				}
				*b_1_begin++ = sum;
				V_0_begin -= N;
			}
			V_0_begin += N;
		}
	}
	std::cout << "D : ";

	TIME_CHECK

	b_1_.fill(0);

	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	auto V_0_end = V_0_.end();
	auto W_1_end = W_1_.end();
	auto b_1_end = b_1_.end();

	using W_1_it = decltype(W_1_begin);
	using V_0_it = decltype(V_0_begin);
	using b_1_it = decltype(b_1_begin);

	b_1_.fill(0);

	b_1_id = 0;
	V_0_id = 0;
	W_1_id = 0;
	for (int k = 0; k < mB; k++) {
		W_1_it w1It = W_1_begin;
		for (b_1_it b1It = b_1_begin; b1It != b_1_begin + M; ++b1It) {
			float sum = 0;
			for (V_0_it v0It = V_0_begin; v0It != V_0_begin + N; ++v0It) {
//				std::cout << "...b[" << b_1_id << "] += W[" << W_1_id++ << "] * V[" << V_0_id++ << "]" << std::endl;
				const float w1 = *w1It++;
				const float v0 = *v0It;
				sum += w1 * v0;
			}
			*b1It = sum;
			//V_0_begin -= N;

			b_1_id++;
			V_0_id -= N;
		}
		b_1_begin += M;
		V_0_begin += N;
		V_0_id += N;
		W_1_id -= N * M;

	}
	V_0_begin -= N * mB;
	b_1_begin -= M * mB;

	std::cout << std::distance(V_0_begin, V_0_.begin()) << std::endl;


	b_1_begin = b_1_.begin();
	b_1_id = 0;
	for (int k = 0; k < mB; k++) {
		for (int j = 0; j < M; j++) {
//			std::cout << "...b_B[" << k * M + j << "] = " << b_1_begin[b_1_id++] << std::endl;

		}
	}

	b_1_.fill(0);

	TIME_START


	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();

	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;
		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			for (b_1_it b1It = b_1_begin + M * k; b1It != b_1_begin + M * (k + 1); ++b1It) {
				for (V_0_it v0It = V_0_begin + N * k; v0It != V_0_begin + N * (k + 1); ++v0It, ++w1It) {
					*b1It += *w1It * *v0It;
				}
			}
		}
	}

	std::cout << "E : ";
	TIME_CHECK

	b_1_.fill(0);
	TIME_START
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();


	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;

		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			for (b_1_it b1It = b_1_begin + (M * k); b1It != b_1_begin + (M * (k + 1)); ++b1It) {
				*b1It = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
				w1It += N;
			}
		}
	}


	std::cout << "G : ";
	TIME_CHECK
	b_1_.fill(0);

	TIME_START
	W_1_begin = W_1_.begin();
	V_0_begin = V_0_.begin();
	b_1_begin = b_1_.begin();

	V_0_begin_ = V_0_.begin();
	b_1_begin_ = b_1_.begin();


	for (int d = 0; d < TRIALS; d++) {
		V_0_begin = V_0_begin_;
		b_1_begin = b_1_begin_;

		for (int k = 0; k < mB; k++) {
			W_1_it w1It = W_1_begin;
			std::generate(b_1_begin + (M * k), b_1_begin + (M * (k + 1)), [&V_0_begin, &w1It, k] {
				auto o = std::inner_product(V_0_begin + (N * k), V_0_begin + (N * (k + 1)), w1It, float{0});
				w1It += N;
				return o;

			});
		}
	}


	std::cout << "H : ";
	TIME_CHECK

	//TODO COPY FROM TEST_SMALL IN HERE FOR BIG ARRAYS TEST

	return 0;

}


int main() {

	std::cout << "TEST SMALL" << std::endl;
	test_small();
	std::cout << "TEST BIG" << std::endl;

	test_big();

	return 0;
}
