# MSBD5009 Parallel Computing Notes

## Content
- [Concept]()
- [MPI](#mpi)
- [pThread]()
- [OpenMP]()
- [CUDA]()

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
```

## pThread
```c++

```


## CUDA
```c++

```
