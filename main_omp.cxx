#include <stdio.h>
#include <cstdlib>
#include <omp.h>
//#include <mpi.h>
#include <inttypes.h>

#include "shared.h"

uint64_t compute_cycle(region_t* src_reg, region_t* dst_reg, int dist){
    uint64_t sum = 0; 
    uint64_t num_alive = 0;
    #pragma omp parallel for reduction(+: sum, num_alive) default(shared) 
    for(size_t x = dist; x < src_reg->w - dist; x++){
        //int tn = omp_get_thread_num();
        //printf("Finished slice with thread num %d\n", tn);
        uint64_t loc_sum = 0;
        uint64_t loc_num_alive = 0;
        for(size_t y = dist; y < src_reg->h - dist; y++){
            for(size_t z = dist; z < src_reg->d - dist; z++){
                loc_num_alive += read_cell(src_reg, x, y, z);
                loc_sum += apply_rule(src_reg, dst_reg, x, y, z, dist);
            }
        }
        sum += loc_sum;
        num_alive += loc_num_alive;
        //printf("local sum: %lu, local_num_alive: %lu\n", loc_sum, loc_num_alive);
    }
    #pragma omp barrier
    printf("Alive in input: %lu\n", num_alive);
    return sum;
}

void swap_regions(region_t** reg_1, region_t** reg_2){
    region_t* tmp_a = *reg_1;
    region_t* tmp_b = *reg_2;
    *reg_1 = tmp_b;
    *reg_2 = tmp_a;
}


int main(int argc, char** argv){
    //args: edge length, iterations
    int edge_len = 0;
    int iters = 0;
    if(argc != 3){
        printf("Usage: <edge length> <num iterations>\n");
        return 1;
    }
    else {
        edge_len = atoi(argv[1]);
        iters = atoi(argv[2]);
        printf("Running %d iters with edge length: %d\n", iters, edge_len);
    }
    int num_threads = omp_get_max_threads();
    printf("NUM THREADS %d\n", num_threads);

    int expanded_len = edge_len + (2 * DISTANCE);

    region_t* reg_a = init_region(expanded_len, expanded_len, expanded_len);
    region_t* reg_b = init_region(expanded_len, expanded_len, expanded_len);

    box_bounds_t bbox = box_bounds_t{
        .x_min = X_MIN + DISTANCE,
        .x_max = X_MAX + DISTANCE,
        .y_min = Y_MIN + DISTANCE,
        .y_max = Y_MAX + DISTANCE,
        .z_min = Z_MIN + DISTANCE,
        .z_max = Z_MAX + DISTANCE,
    };

    fill_region(reg_a, &bbox);

    for(size_t i = 0; i < iters; i++){
        printf("computing cycle %d\n", i + 1);
        auto t0 = Time::now();
        uint64_t changed = compute_cycle(reg_a, reg_b, DISTANCE);
        swap_regions(&reg_a, &reg_b);
        auto t1 = Time::now();
        fsec_t fs = t1 - t0;
        ms_t dur = std::chrono::duration_cast<ms_t>(fs);
        printf("iter %d: %lu cells updated in %dms\n", i + 1, changed, dur.count());

    }
}