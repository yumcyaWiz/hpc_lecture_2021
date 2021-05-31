#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace std;

__global__ void matmul(float* A, float* B, float* C, int M, int N, int offset) {
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
  C[M * i + j + offset] = sum;
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

  int N = 4096;
  int M = N / size;
  int blockSize = 1024;
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
  cudaMallocManaged(&subB, N * M * sizeof(float));
  cudaMallocManaged(&subC, M * N * sizeof(float));
  vector<float> recv(N * M);
  for (int i = 0; i < M * N; i++) {
    subC[i] = 0;
  }

  int offset = N / size * rank;
#pragma omp parallel for
  for (int i = 0; i < N / size; i++) {
    for (int j = 0; j < N; j++) {
      subA[N * i + j] = A[N * (i + offset) + j];
    }
  }
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N / size; j++) {
      subB[M * i + j] = B[N * i + j + offset];
    }
  }

  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  double comp_time = 0, comm_time = 0;
  for (int irank = 0; irank < size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset = M * ((rank + irank) % size);

    dim3 grid(M / blockSize, M);
    matmul<<<grid, blockSize, blockSize * sizeof(float)>>>(subA, subB, subC, M,
                                                           N, offset);
    cudaDeviceSynchronize();
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();

    MPI_Request request[2];
    MPI_Isend(&subB[0], N * N / size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD,
              &request[0]);
    MPI_Irecv(&recv[0], N * N / size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD,
              &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
#pragma omp parallel for
    for (int i = 0; i < N * N / size; i++) {
      subB[i] = recv[i];
    }
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  MPI_Allgather(&subC[0], N * N / size, MPI_FLOAT, &C[0], N * N / size,
                MPI_FLOAT, MPI_COMM_WORLD);

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