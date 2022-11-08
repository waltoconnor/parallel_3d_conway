# 3D Conway's Game of Life
These are parallelized implementations of 3 dimensional Conway's game of life, one with OpenMP, and the other with OpenMPI.
This was built as part of a class project to evaluate the overheads of the cross machine communication that comes with OpenMPI for a relatively simple task.

The OpenMPI version was implemented by slicing the workspace in to `n` slices along the x axis (as opposed to cubes), with one slice per available thread. 

Interestingly the OpenMPI version performed substantially better, likely due to the much improved memory locality of the smaller slices (these were running on multi CPU machines, so the OpenMPI version likely incurred brutal NUMA latency penalties).

For more information, see the report pdf.