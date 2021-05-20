// mpic++ -std=c++11 clustering_mpi_skeleton.cpp clustering_impl.cpp -o clustering
// mpiexec -n 4 ./clustering ./dataset/test1 ./results/
// mpic++ -std=c++11 clustering_mpi_skeleton_final1.cpp clustering_impl.cpp -o clustering; mpiexec -n 4 ./clustering ./dataset/test2 ./results/
// mpic++ -std=c++11 clustering_mpi_skeleton_final1.cpp clustering_impl.cpp -o clustering; mpiexec -n 4 ./clustering ./dataset/test1 ./results/


#include "clustering.h"

#include "mpi.h"

#include <cassert>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm;
  int num_process; // number of processors
  int my_rank;     // my global rank

  comm = MPI_COMM_WORLD;

  MPI_Comm_size(comm, &num_process);
  MPI_Comm_rank(comm, &my_rank);

  if (argc != 3) {
    std::cerr << "usage: ./clustering_sequential data_path result_path"
              << std::endl;

    return -1;
  }
  std::string dir(argv[1]);
  std::string result_path(argv[2]);

  int num_graphs;
  int *clustering_results = nullptr;
  int *num_cluster_total = nullptr;

  int *nbr_offs = nullptr, *nbrs = nullptr;
  int *nbr_offs_local = nullptr, *nbrs_local = nullptr;

  GraphMetaInfo *info = nullptr;

  // read graph info from files
  if (my_rank == 0) {
    num_graphs = read_files(dir, info, nbr_offs, nbrs);
  }
  auto start_clock = chrono::high_resolution_clock::now();

  // ADD THE CODE HERE===============================
  /*
    Env Setting
  */
  
  MPI_Bcast(&num_graphs, 1, MPI_INT, 0, comm);
  int scatter_count = num_graphs/num_process;

  /*
    This piece of code allow you to create your own datatype in MPI
  */
  MPI_Datatype mpi_dt_GraphMetaInfo;
  MPI_Type_contiguous(2, MPI_INT, &mpi_dt_GraphMetaInfo);
  MPI_Type_commit(&mpi_dt_GraphMetaInfo);
  
  //==========for parallel==================

  GraphMetaInfo *info_scattered = nullptr;
  info_scattered = (GraphMetaInfo *)calloc(scatter_count, sizeof(GraphMetaInfo));

  MPI_Scatter(info, scatter_count, mpi_dt_GraphMetaInfo,
      info_scattered, scatter_count, mpi_dt_GraphMetaInfo,
      0, comm);


  // nbr distribution with pointer and MPI Send recv
  if (my_rank == 0) {
    MPI_Request request;
    for (int q = 0; q < num_process; q++) {
      int info_num_edges_shift = 0; //reset everytime
      int info_num_vertices_shift = 0;
      static int * nbrs_temp = nbrs; //preserve the previous value
      static int * nbr_offs_temp = nbr_offs;

      for (int i = 0; i < scatter_count; i++){
        info_num_edges_shift = info_num_edges_shift + info[i+q*scatter_count].num_edges + 1;
        info_num_vertices_shift = info_num_vertices_shift + info[i+q*scatter_count].num_vertices + 1;
      }
     
      MPI_Isend(nbrs_temp, info_num_edges_shift, MPI_INT, q, q+10,  MPI_COMM_WORLD, &request); 
      MPI_Isend(nbr_offs_temp, info_num_vertices_shift, MPI_INT, q, q+20,  MPI_COMM_WORLD, &request); 
      // MPI_Wait(&request, MPI_STATUS_IGNORE);
      // cout << "rank: " << my_rank<< " : nbrs sent" << endl;
      nbrs_temp += info_num_edges_shift;
      nbr_offs_temp += info_num_vertices_shift;
    }
    
  }
  // cout << "rank: " << my_rank<< " : MPI_Barrier1" << endl;
  // MPI_Barrier(comm);
  // cout << "rank: " << my_rank<< " : MPI_Barrier2" << endl;
  //receving the nbr & nbr offset
  int info_num_edges_shift = 0;
  int info_num_vertices_shift = 0;
  for (int i = 0; i < scatter_count; i++){
    info_num_edges_shift = info_num_edges_shift + info_scattered[i].num_edges + 1;
    info_num_vertices_shift = info_num_vertices_shift + info_scattered[i].num_vertices + 1;
  }
  nbrs_local = (int *)calloc(info_num_edges_shift, sizeof(int));
  nbr_offs_local = (int *)calloc(info_num_vertices_shift, sizeof(int));
  MPI_Recv(nbrs_local, info_num_edges_shift, MPI_INT, 0, my_rank+10,  MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
  MPI_Recv(nbr_offs_local, info_num_vertices_shift, MPI_INT, 0, my_rank+20,  MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
  // cout << "rank: " << my_rank<< " : nbrs recv" << endl;
  // cluster index for each vertices
  int *clustering_results_scattered[scatter_count];
  // number of clusters for each graph
  int num_cluster_total_scattered[scatter_count];

  // MPI_Barrier(comm);

  int info_local_num_vertices_scattered_total = 0;
  
  for (size_t i = 0; i < scatter_count; i++) {
    MPI_Request mpi_request;
    GraphMetaInfo info_local = info_scattered[i];
    clustering_results_scattered[i] =
        (int *)calloc(info_local.num_vertices, sizeof(int));
    info_local_num_vertices_scattered_total += info_local.num_vertices;

    int num_cluster_local = clustering(info_local, nbr_offs_local, nbrs_local,
                                       clustering_results_scattered[i]);

    // for (int ii = 0; ii < info_local.num_vertices; ii++){
    //   cout << clustering_results_scattered[i][ii] << " ";
    // }
    // cout << endl;
    MPI_Isend(clustering_results_scattered[i], info_local.num_vertices, MPI_INT, 0, my_rank*100+i,  MPI_COMM_WORLD, &mpi_request); 
    // cout << "rank: " << my_rank<< " :clustering_results sent" << endl;
    num_cluster_total_scattered[i] = num_cluster_local;
    // printf("num cluster in graph %d : %d\n", i, num_cluster_local);
    
    nbr_offs_local += (info_local.num_vertices + 1);
    nbrs_local += (info_local.num_edges + 1);
    // MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
  }

  // cout << "rank: " << my_rank<< " : MPI_Barrier 146" << endl;
  // MPI_Barrier(MPI_COMM_WORLD);
  // cout << "rank: " << my_rank<< " : MPI_Barrier 147" << endl;

  //Receive the clustering results_scatter back
  if (my_rank == 0){
    int sumofnum_vertices = 0;
    for (int i = 0; i < num_graphs; i++){
      sumofnum_vertices += info[i].num_vertices;
    }
    clustering_results = (int *)calloc(sumofnum_vertices, sizeof(int));
    int* temp = clustering_results;

    for (int q = 0; q < num_process; q++){
      for(int i = 0; i < scatter_count; i++){
        MPI_Recv(temp, info[q*scatter_count+i].num_vertices, MPI_INT, q, q*100+i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // cout << temp[0] << " " << endl;
        temp += info[q*scatter_count+i].num_vertices;
      }
    }
  }
 
  //Gather the num_cluster_total here
  if (my_rank == 0){
    num_cluster_total = (int *)calloc(num_graphs, sizeof(int));
  }
  // MPI_Barrier(comm);
  MPI_Gather(num_cluster_total_scattered, scatter_count, MPI_INT,
        num_cluster_total, scatter_count, MPI_INT,
        0, MPI_COMM_WORLD);



  // ADD THE CODE HERE===============================

  MPI_Barrier(comm);
  auto end_clock = chrono::high_resolution_clock::now();

  // 1) print results to screen
  if (my_rank == 0) {
    for (size_t i = 0; i < num_graphs; i++) {
      printf("num cluster in graph %d : %d\n", i, num_cluster_total[i]);
    }
    fprintf(stderr, "Elapsed Time: %.9lf ms\n",
            chrono::duration_cast<chrono::nanoseconds>(end_clock - start_clock)
                    .count() /
                pow(10, 6));
  }

  // 2) write results to file
  if (my_rank == 0) {
    int *result_graph = clustering_results;
    for (int i = 0; i < num_graphs; i++) {
      GraphMetaInfo info_local = info[i];
      write_result_to_file(info_local, i, num_cluster_total[i], result_graph,
                           result_path);

      result_graph += info_local.num_vertices;
    }
  }

  MPI_Finalize();

  if (my_rank == 0) {
    free(num_cluster_total);
  }

  return 0;
}
