/*
 * Name: Wong Tsz Ho
 * Student id: 20725187
 * ITSC email: thwongbi@connect.ust.hk
 *
 * Please only change this file and do not change any other files.
 * Feel free to change/add any helper functions.
 *
 * COMPILE: g++ -lstdc++ -std=c++11 -lpthread clustering_pthread_skeleton.cpp -main.cpp -o pthread
 * RUN:     ./pthread <path> <epsilon> <mu> <num_threads>
 */

// g++ -lstdc++ -std=c++11 -pthread clustering_pthread_skeleton01_done.cpp main.cpp -o pthread; ./pthread ./datasets/test2/0.txt 0.7 3 2
// g++ -lstdc++ -std=c++11 -pthread clustering_pthread_skeleton01_done.cpp main.cpp -o pthread; ./pthread ./datasets/test3/com-lj-500m.txt 0.7 3 2


#include <pthread.h>
#include "clustering.h"

struct AllThings{
    int num_threads;
    int my_rank;

    //var that needed to pass to the thread
    float epsilon;
    int mu;
    int num_vs;
    int num_es;

    int *nbr_offs;
    int *nbrs;

    bool *pivots;
    int *num_sim_nbrs;
    int **sim_nbrs;


    AllThings(  int inum_threads, 
                int imy_rank,
                float iepsilon,
                int imu,
                int inum_vs,
                int inum_es,
                int *inbr_offs,
                int *inbrs,
                bool *ipivots,
                int *inum_sim_nbrs,
                int **isim_nbrs

                ){
        num_threads = inum_threads;
        my_rank = imy_rank;

        epsilon = iepsilon;
        mu = imu;
        num_vs = inum_vs;
        num_es = inum_es;

        nbr_offs = inbr_offs;
        nbrs = inbrs;

        pivots = ipivots;
        num_sim_nbrs = inum_sim_nbrs;
        sim_nbrs = isim_nbrs;
    };
};

void *parallel(void* allthings){
    AllThings *all = (AllThings *) allthings;

    for (int i = 0; i < all->num_vs; i++){
        if (all->my_rank == (i % all->num_threads)){
            int *left_start = &all->nbrs[all->nbr_offs[i]];
            int *left_end = &all->nbrs[all->nbr_offs[i + 1]];
            int left_size = left_end - left_start;

            all->sim_nbrs[i] = new int[left_size];
            // loop over all neighbors of i
            for (int *j = left_start; j < left_end; j++) {
                int nbr_id = *j;

                int *right_start = &all->nbrs[all->nbr_offs[nbr_id]];
                int *right_end = &all->nbrs[all->nbr_offs[nbr_id + 1]];
                int right_size = right_end - right_start;

                // compute the similarity
                int num_com_nbrs = get_num_com_nbrs(left_start, left_end, right_start, right_end);

                float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

                if (sim > all->epsilon) {
                    all->sim_nbrs[i][all->num_sim_nbrs[i]] = nbr_id;
                    all->num_sim_nbrs[i]++;
                }
            }
            if (all->num_sim_nbrs[i] > all->mu) all->pivots[i] = true;
            // cout << all->num_sim_nbrs[i] << " : ";
        }
    }
    // printf("Hello from %d of %d\n", all->my_rank, all->num_threads);


    return 0;
}


//this not yet parallel
void expansion(int cur_id, int num_clusters, int *num_sim_nbrs, int **sim_nbrs,
               bool *visited, bool *pivots, int *cluster_result) {
  for (int i = 0; i < num_sim_nbrs[cur_id]; i++) {
    int nbr_id = sim_nbrs[cur_id][i];
    if ((pivots[nbr_id])&&(!visited[nbr_id])){
      visited[nbr_id] = true;
      cluster_result[nbr_id] = num_clusters;
      expansion(nbr_id, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots,
                cluster_result);
    }
  }
}

int *scan(float epsilon, int mu, int num_threads, int num_vs, int num_es, int *nbr_offs, int *nbrs){
    long thread;
    pthread_t* thread_handles = (pthread_t*) malloc(num_threads*sizeof(pthread_t));
    int *cluster_result = new int[num_vs];

    bool *pivots = new bool[num_vs]();
    int *num_sim_nbrs = new int[num_vs]();
    int **sim_nbrs = new int*[num_vs];


    
    for (thread=0; thread < num_threads; thread++){
        pthread_create(&thread_handles[thread], NULL, parallel, 
        (void *) new AllThings( 
            num_threads, 
            thread,
            epsilon,
            mu ,
            num_vs,
            num_es ,
            nbr_offs ,
            nbrs,
            pivots,
            num_sim_nbrs,
            sim_nbrs
        )
        );
    }

    for (thread=0; thread < num_threads; thread++){
        pthread_join(thread_handles[thread], NULL);
    }

    // Stage 2: (sim_nbrs, num_sim_nbrs, pivots)
    bool *visited = new bool[num_vs]();
    std::fill(cluster_result, cluster_result + num_vs, -1);
    int num_clusters = 0;
    for (int i = 0; i < num_vs; i++) {
        if (!pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i, num_sim_nbrs, sim_nbrs, visited, pivots, cluster_result);

        num_clusters++;
    }


    delete[] pivots;
    delete[] num_sim_nbrs;
    delete[] sim_nbrs;
    delete[] visited;

    return cluster_result;
}



