# MSBD5009 Parallel Computing Notes

## Content
- [Concept]()
- [MPI](#mpi)
- [pThread](#pthread)
- [OpenMP]()
- [CUDA](#cuda)

## MPI
Important component
```C++
// MPI initialization
MPI_Init(&argc, &argv);

// set `p` and `my_rank`
MPI_Comm_size(comm, &p);
MPI_Comm_rank(comm, &my_rank);

// wait for all processes to finish
MPI_Barrier(comm);

## Application Based
// each process works on its portion of the vertices
for (int v = my_begin_offset; v < my_end_offset; v++) {

// update (reduce) `is_path_found` among all processes
 MPI_Allreduce(MPI_IN_PLACE, &is_path_found, 1, MPI_C_BOOL, MPI_LOR, comm);
 
 // broadcast process i’s data in `in_queue_next`
 MPI_Bcast(in_queue_next + iter_v_beg_tmp, iter_v_end_tmp - iter_v_beg_tmp, MPI_C_BOOL, i, comm);
 
 // broadcast process i’s data in `parent`
 MPI_Bcast(parent + iter_v_beg, iter_v_end - iter_v_beg, MPI_INT, i, comm);
 
  // broadcast `loc_n`, `loc_src` and `loc_sink`
 MPI_Bcast(&loc_n, 1, MPI_INT, 0, comm);
 MPI_Bcast(&loc_src, 1, MPI_INT, 0, comm);
 MPI_Bcast(&loc_sink, 1, MPI_INT, 0, comm);
 
  // broadcast loc_cap, loc_flow
 MPI_Bcast(loc_cap, loc_n * loc_n, MPI_INT, 0, comm);
 MPI_Bcast(loc_flow, loc_n * loc_n, MPI_INT, 0, comm)
 
// copy results back to array `flow`
 if (my_rank == 0) {
 memcpy(flow, loc_flow, sizeof(int) * loc_n * loc_n);
 }
 
 //**********************
 // MPI_initialization
 MPI_Init(&argc, &argv);
 
 //set num_machines and my_rank
 MPI_Comm_size(comm, &num_machines);
 MPI_Comm_rank(comm, &my_rank);
 
//broadcast ref_len, read_len, and num_reads
MPI_Bcast(&ref_len, 1, MPI_INT, 0, comm);
MPI_Bcast(&read_len, 1, MPI_INT, 0, comm);
MPI_Bcast(&num_reads, 1, MPI_INT, 0, comm);

//Other processes allocate the reference string
if (my_rank > 0){
 ref = (char *)malloc(sizeof(char) * (ref_len + 1));
}

//broadcast the reference string
MPI_Bcast(ref, ref_len+1, MPI_CHAR, 0, comm);

//number of reads to align in current process
int my_num_reads = offsets[my_rank + 1] - offsets[my_rank];

//process 0 sends reads to other processes
if(my_rank == 0){
 for (int i = 1; i < num_machines; i++){
  MPI_Send(reads + offsets[i] * (read_len+1),(offsets[i+1]-offsets[i]) * (read_len + 1), MPI_CHAR, i, 0, comm);
 }
}else {
 //other processes allocate memory for reads and receive
 reads = (char *)malloc(sizeof(char) * my_num_reads * (read_len + 1));
 MPI_Recv(reads, my_num_reads * (read_len + 1), MPI_CHAR, 0, 0, comm, MPI_STATUS_IGNORE);
}

if (my_rank == 0) {
 // process 0 receives results from other processes
    for (int i = 1; i < num_machines; i++) {
        MPI_Recv(results + offsets[i], offsets[i + 1] - offsets[i], MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
    }
    // output results
    for (int i = 0; i < num_reads; i++) {
        cout << results[i] << endl;
    }
    } else {
    // other processes send local results to process 0
        MPI_Send(results, my_num_reads, MPI_INT, 0, 0, comm);
    }
```


## pThread
```c++
// synchronize all threads
 pthread_barrier_wait(&barrier);
 
 // critical area to update `in_queue2` and its length
 pthread_mutex_lock(&mutex);
 in_queue2[in_queue2_len] = v;
 in_queue2_len++;
 pthread_mutex_unlock(&mutex);
 
 // synchronize all threads
pthread_barrier_wait(&barrier);

 // initialize `mutex` and `barrier`
 pthread_mutex_init(&mutex, nullptr);
 pthread_barrier_init(&barrier, nullptr, num_threads);
 
 // create threads to run
pthread_create(&threads[i], &attr, thread_work, (void *)&parameters[i]);

 // wait for all threads to finish
 for (int i = 0; i < num_threads; i++) {
 pthread_join(threads[i], nullptr);
 }
 
 //**********************
// critical section starts before updating read_id and next read_id
pthread_mutex_lock(&mutex);
// end of critical section
pthread_mutex_unlock(&mutex);

//exit if all reads has been aligned
if(read_id >= my_num_reads){
 break;;
}

//initialize mutex
pthread_mutex_init(&mutex, nullptr);

//create_threads
pthread_create(&threads[i], &attr, worker, (void *)&params[i]);

//wait for all threads to finish
for (int i = 0; i < num_threads; i++){
 pthread_join(threads[i], nullptr);
}
```


## CUDA
```c++
int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
int element_skip = blockDim.x * gridDim.x;
// each thread iterates over `u` with distance of `element_skip`
for (auto u = global_tid; u < N; u += element_skip) {

// use one thread to set `d_is_path_found` to false
if (global_tid == 0) {
 *d_is_path_found = false;
}

// each thread iterates over `v` with distance of `element_skip`
for (int v = global_tid; v < N; v += element_skip) {

// invoke kernel `init_bfs_structures` to initialize BFS related data structures
init_bfs_structures<<<blocks, threads>>>(src, N, d_visited, d_in_queue_even, d_in_queue_odd, d_is_path_found);
 
// invoke kernel bfs_one_depth to compute single BFS iteration
bfs_one_depth<<<blocks, threads>>>(sink, N, d_visited, d_in_enqueue, d_in_enqueue_next, d_is_path_found, d_cap, d_flow, d_parent);

// allocate `d_cap`,`d_flow`,`d_visited`,`d_in_queue_even`,`d_in_queue_odd`,`d_parent` and `d_is_path_found` on CUDA device
cudaMalloc(&d_cap, sizeof(int) * N * N);
cudaMalloc(&d_flow, sizeof(int) * N * N);
cudaMalloc(&d_visited, sizeof(bool) * N);
cudaMalloc(&d_in_queue_even, sizeof(bool) * N);
cudaMalloc(&d_in_queue_odd, sizeof(bool) * N);
cudaMalloc(&d_parent, sizeof(int) * N);
cudaMalloc(&d_is_path_found, sizeof(bool));

// copy `flow` and `cap` to device
cudaMemcpy(d_flow, flow, sizeof(int) * N * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_cap, cap, sizeof(int) * N * N, cudaMemcpyHostToDevice);

// copy `parent` to host
cudaMemcpy(parent, d_parent, sizeof(int) * N, cudaMemcpyDeviceToHost);

// update flow to device
cudaMemcpy(d_flow, flow, sizeof(int) * N * N, cudaMemcpyHostToDevice);

//**********************
//exit if current thread has no read to align
if(tid >= d_num_reads){
 return;
}

// cllocate memory on GPU
cudaMalloc(&d_ref, sizeof(char) * (ref_len + 1)));
// copy reference and reads to GPU
cudaMemcpy(d_ref, ref, sizeof(char) * (ref_len + 1), cudaMemcpyHostToDevice);
// copy results to CPU
cudaMemcpy(results + read_id, d_results, sizeof(int) * d_num_reads, cudaMemcpyDeviceToHost);
```
