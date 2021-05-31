#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
using namespace std;

__global__ void matmul(float *A, float *B, float *C, int M, int N) {
  int i = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.0f;
  extern __shared__ float A_s[];
  for (int ks = 0; ks < N; ks += blockDim.x) {
    __syncthreads();
    A_s[threadIdx.x] = A[N * i + ks + threadIdx.x];
    __syncthreads();
    for (int k = ks; k < ks + blockDim.x; k++) {
      sum += A_s[k - ks] * B[M * k + j];
    }
  }
  C[M * i + j] = sum;
}

int main(int argc, char **argv) {
  int M = 1024;
  int N = 2048;
  int blockSize = 1024;
  float *A, *B, *C;
  cudaMallocManaged(&A, M * N * sizeof(float));
  cudaMallocManaged(&B, N * M * sizeof(float));
  cudaMallocManaged(&C, M * M * sizeof(float));

  int gpusize, gpurank;
  cudaGetDeviceCount(&gpusize);
  cudaSetDevice(1);
  cudaGetDevice(&gpurank);
  printf("gpusize: %d\ngpurank: %d\n", gpusize, gpurank);

  std::random_device seed_gen;
  std::mt19937 mt(seed_gen());
  std::uniform_real_distribution<float> dist(0, 1);
#pragma omp parallel for collapse(2) private(mt, dist)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A[N * i + j] = dist(mt);
    }
  }
#pragma omp parallel for collapse(2) private(mt, dist)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      B[M * i + j] = dist(mt);
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      C[M * i + j] = 0;
    }
  }

  dim3 grid(M / blockSize, M);
  auto tic = chrono::steady_clock::now();
  matmul<<<grid, blockSize, blockSize * sizeof(float)>>>(A, B, C, M, N);
  cudaDeviceSynchronize();
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n", N, time, 2. * M * N * N / time / 1e9);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < M; j++) {
        C[M * i + j] -= A[N * i + k] * B[M * k + j];
      }
    }
  }
  double err = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      err += fabs(C[M * i + j]);
    }
  }
  printf("error: %lf\n", err / M / M);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
