#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace std;

__global__ void matmul(float* A, float* B, float* C, int N, int offset) {
  int i = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.0f;
  extern __shared__ float A_s[];
  for (int ks = 0; ks < N; ks += blockDim.x) {
    __syncthreads();
    A_s[threadIdx.x] = A[N * i + ks + threadIdx.x];
    __syncthreads();
    for (int k = ks; k < ks + blockDim.x; k++) {
      sum += A_s[k - ks] * B[N * k + j];
    }
  }
  C[N * i + j + offset] = sum;
}

int main(int argc, char** argv) {
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int gpusize, gpurank;
  cudaGetDeviceCount(&gpusize);
  cudaSetDevice(rank % gpusize);
  cudaGetDevice(&gpurank);

  int N = 1024;
  int M = N / size;
  int blockSize = 128;
  vector<float> A(N * N);
  vector<float> B(N * N);
  vector<float> C(N * N, 0);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[N * i + j] = drand48();
      B[N * i + j] = drand48();
    }
  }

  float* subA;
  float* subB;
  float* subC;
  cudaMallocManaged(&subA, M * N * sizeof(float));
  cudaMallocManaged(&subB, N * N * sizeof(float));
  cudaMallocManaged(&subC, M * N * sizeof(float));
  for (int i = 0; i < M * N; i++) {
    subC[i] = 0;
  }

  int offset = M * rank;
#pragma omp parallel for
  for (int i = 0; i < N / size; i++) {
    for (int j = 0; j < N; j++) {
      subA[N * i + j] = A[N * (i + offset) + j];
    }
  }
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      subB[N * i + j] = B[N * i + j];
    }
  }

  double comp_time = 0, comm_time = 0;
  auto tic = chrono::steady_clock::now();
  offset = M * N * rank;

  dim3 grid(N / blockSize, M);
  matmul<<<grid, blockSize, blockSize * sizeof(float)>>>(subA, subB, subC, N,
                                                         offset);
  cudaDeviceSynchronize();
  auto toc = chrono::steady_clock::now();
  comp_time += chrono::duration<double>(toc - tic).count();

  tic = chrono::steady_clock::now();
  comm_time += chrono::duration<double>(tic - toc).count();

  MPI_Allgather(&subC[0], M * N, MPI_FLOAT, &C[0], M * N, MPI_FLOAT,
                MPI_COMM_WORLD);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[N * i + j] -= A[N * i + k] * B[N * k + j];
      }
    }
  }
  double err = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      err += fabs(C[N * i + j]);
    }
  }
  if (rank == 0) {
    double time = comp_time + comm_time;
    printf("N    : %d\n", N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n", time, 2. * N * N * N / time / 1e9);
    printf("error: %lf\n", err / N / N);
  }

  cudaFree(subA);
  cudaFree(subB);
  cudaFree(subC);

  MPI_Finalize();
}