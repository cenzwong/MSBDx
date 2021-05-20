/* 
 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp main.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>

  id: 20725187
  name: Wong, Tsz Ho
  email: thwongbi@connect.ust.hk


   nvcc -std=c++11 clustering_cuda_skeleton_done2.cu clustering_impl.cpp main.cpp -o cuda; ./cuda ./datasets/test2/0.txt 0.7 3 8 512
   nvcc -std=c++11 clustering_cuda_skeleton_done2.cu clustering_impl.cpp main.cpp -o cuda; ./cuda ./datasets/cuda-test/0.txt 0.7 3 8 512

 */

#include <iostream>
#include "clustering.h"
#include <algorithm>

__global__ void stage1(
  float iepsilon,
  int imu,
  int inum_vs,
  int inum_es,
  int *inbr_offs,
  int *inbrs,
  bool *ipivots,
  int *inum_sim_nbrs,
  int *isim_nbrs
);

__device__ int get_num_com_nbrs(int *nbrs_left_start, int *nbrs_left_end, int *nbrs_right_start, int *nbrs_right_end);
// Define variables or functions here

__host__ 
void expansion(int cur_id, int num_clusters, int *num_sim_nbrs, int *sim_nbrs,
  bool *visited, bool *pivots, int *cluster_result, int* nbr_offs) {
    for (int i = 0; i < num_sim_nbrs[cur_id]; i++) {
      int nbr_id = sim_nbrs[nbr_offs[cur_id] + i];
      if ((pivots[nbr_id])&&(!visited[nbr_id])){
        visited[nbr_id] = true;
        cluster_result[nbr_id] = num_clusters;
        expansion(nbr_id, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots,
          cluster_result, nbr_offs);
      }
    }
}

__device__ 
int get_num_com_nbrs(int *nbrs_left_start, int *nbrs_left_end, int *nbrs_right_start, int *nbrs_right_end) {
  int *nbrs_left_pos = nbrs_left_start, *nbrs_right_pos = nbrs_right_start, num_com_nbrs = 0;

  while (nbrs_left_pos < nbrs_left_end && nbrs_right_pos < nbrs_right_end) {
      if (*nbrs_left_pos == *nbrs_right_pos) {
        num_com_nbrs++;
        nbrs_left_pos++;
        nbrs_right_pos++;
      } else if (*nbrs_left_pos < *nbrs_right_pos) {
        nbrs_left_pos++;
      } else {
        nbrs_right_pos++;
      }
  }
  return num_com_nbrs;
}



// __host__  
// void getMaxLeftSize(int num_vs, int *nbr_offs, int &max_left_size){
//   int temp_max = 0;
//   for (int i = 0; i < num_vs; i++) {
//     int left_start = nbr_offs[i];
//     int left_end = nbr_offs[i + 1];
//     int left_size = left_end - left_start;

//     if (left_size > temp_max){
//       temp_max = left_size;
//     }
//   }
//   max_left_size = temp_max;
// }

__global__ 
void stage1(
  float epsilon,
  int mu,
  int num_vs,
  int num_es,
  int *nbr_offs,
  int *nbrs,
  bool *pivots,
  int *num_sim_nbrs,
  int *sim_nbrs
){
  // // printf("__epsilon: %f \r\n", epsilon);
  // // printf("__mu: %d \r\n", mu);
  // printf("__num_vs: %d \r\n", num_vs);
  // printf("__num_es: %d \r\n", num_es);

  // // int* ptr nullptr;
  // // ptr =  cudaGetSymbolAddress(__cuda__nbr_offs);
  // // printf("__cuda__nbr_offs: %d %d %d %d %d \r\n", ptr[0], ptr[1], ptr[2], ptr[num_vs], ptr[num_vs + 1]);
  // printf("__cuda__nbr_offs: %d %d %d %d %d \r\n", nbr_offs[0], nbr_offs[1], nbr_offs[2], nbr_offs[num_vs], nbr_offs[num_vs + 1]);
  // printf("__nbr: %d %d %d %d %d \r\n", nbrs[0], nbrs[1], nbrs[2], nbrs[num_es], nbrs[num_es + 1]);
  // printf("__pivots: %d %d %d %d %d \r\n", pivots[0], pivots[1], pivots[2], pivots[num_vs-1], pivots[num_vs]);
  // printf("__num_sim_nbrs: %d %d %d %d %d \r\n", num_sim_nbrs[0], num_sim_nbrs[1], num_sim_nbrs[2], num_sim_nbrs[num_vs-1], num_sim_nbrs[num_vs]);

  const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
  const int nthread = blockDim.x * gridDim.x;

  for (int i = tid; i < num_vs; i += nthread){
    int *nbrs_left_start = &nbrs[nbr_offs[i]];
    int *nbrs_left_end = &nbrs[nbr_offs[i + 1]];
    int left_size = nbrs_left_end - nbrs_left_start;

    // loop over all neighbors of i
    for (int *j = nbrs_left_start; j < nbrs_left_end; j++) {
      int nbr_id = *j;

      int *nbrs_right_start = &nbrs[nbr_offs[nbr_id]];
      int *nbrs_right_end = &nbrs[nbr_offs[nbr_id + 1]];
      int right_size = nbrs_right_end - nbrs_right_start;

      // compute the similarity
      int num_com_nbrs = get_num_com_nbrs(nbrs_left_start, nbrs_left_end, nbrs_right_start, nbrs_right_end);

      float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

      __syncthreads();  
      if (sim > epsilon) {
          sim_nbrs[nbr_offs[i] + num_sim_nbrs[i]] = nbr_id;
          num_sim_nbrs[i]++;
      }
      __syncthreads();  
    }
    if (num_sim_nbrs[i] > mu) pivots[i] = true;
    __syncthreads();
  }

}




__host__ 
void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Stage 1:
    // Fill in the cuda_scan function here
    bool *pivots = new bool[num_vs]();
    int *num_sim_nbrs = new int[num_vs]();
    int *sim_nbrs = new int[num_es];

    std::fill_n(sim_nbrs, num_es, -1);

    // Allocate vectors in device memory
    int* __cuda__nbr_offs = nullptr;
    int* __cuda__nbrs = nullptr;
    bool* __cuda__pivots = nullptr;
    int* __cuda__num_sim_nbrs = nullptr;
    int* __cuda__sim_nbrs = nullptr;
    
    cudaMalloc(&__cuda__nbr_offs, (num_vs+1) * sizeof(int));
    cudaMemcpy(__cuda__nbr_offs, nbr_offs, (num_vs+1) * sizeof(int), cudaMemcpyHostToDevice); // Copy vectors from host memory to device memory
    cudaMalloc(&__cuda__nbrs, (num_es+1) * sizeof(int));
    cudaMemcpy(__cuda__nbrs, nbrs, (num_es+1) * sizeof(int), cudaMemcpyHostToDevice); // Copy vectors from host memory to device memory
    cudaMalloc(&__cuda__pivots, (num_vs) * sizeof(bool));
    // cudaMemcpy(__cuda__pivots, pivots, (num_vs) * sizeof(bool), cudaMemcpyHostToDevice); // Copy vectors from host memory to device memory
    cudaMalloc(&__cuda__num_sim_nbrs, (num_vs) * sizeof(int));
    // cudaMemcpy(__cuda__num_sim_nbrs, num_sim_nbrs, (num_vs) * sizeof(int), cudaMemcpyHostToDevice); // Copy vectors from host memory to device memory
    cudaMalloc(&__cuda__sim_nbrs, (num_es) * sizeof(int));
    // cudaMemcpy(__cuda__num_sim_nbrs, sim_nbrs, (num_vs) * sizeof(int), cudaMemcpyHostToDevice); // Copy vectors from host memory to device memory


    stage1<<<num_blocks_per_grid,num_threads_per_block>>>(
      epsilon,
      mu ,
      num_vs,
      num_es,
      __cuda__nbr_offs,
      __cuda__nbrs,
      __cuda__pivots,
      __cuda__num_sim_nbrs,
      __cuda__sim_nbrs
    );

    cudaMemcpy(sim_nbrs, __cuda__sim_nbrs, (num_es) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_sim_nbrs, __cuda__num_sim_nbrs, (num_vs) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pivots, __cuda__pivots, (num_vs) * sizeof(bool), cudaMemcpyDeviceToHost);

    // cudaFree(cu_epsilon);
    // Stage 2:
    bool *visited = new bool[num_vs]();
    for (int i = 0; i < num_vs; i++) {
      if (!pivots[i] || visited[i]) continue;

      visited[i] = true;
      cluster_result[i] = i;
      expansion(i, i, num_sim_nbrs, sim_nbrs, visited, pivots, cluster_result, nbr_offs);

      num_clusters++;
    }


    delete[] pivots;
    delete[] num_sim_nbrs;
    delete[] sim_nbrs;
    delete[] visited;
}
