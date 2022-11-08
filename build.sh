mpic++ -march=native -O3 main_mpi.cxx -o mpi_version
g++ -march=native -O3 -fopenmp main_omp.cxx -o omp_version