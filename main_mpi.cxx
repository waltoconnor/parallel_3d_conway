#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include "shared.h"

#define MIN_SURVIVE 4
#define MAX_SURVIVE 15
#define MIN_BORN 8

#define EDGE_LEFT_TAG_BASE 11000
#define EDGE_RIGHT_TAG_BASE 12000
#define UPDATED_TAG 20000

void get_boundary_regions(region_t* src_reg, int depth, region_t* dst_arr_xneg, region_t* dst_arr_xpos){
    uint64_t xneg_alive_count = 0;
    for(int x = depth; x < depth + depth; x++){
        for(int y = 0; y < src_reg->h; y++){
            for(int z = 0; z < src_reg->d; z++){
                uint8_t val = read_cell(src_reg, x, y, z);
                xneg_alive_count += val;
                write_cell(dst_arr_xneg, x - depth, y, z, val);
            }
        }
    }

    uint64_t xpos_alive_count = 0;
    int x_send = 0;
    for(int x = src_reg->w - (2 * depth); x < src_reg->w - depth; x++){
        for(int y = 0; y < src_reg->h; y++){
            for(int z = 0; z < src_reg->d; z++){
                uint8_t val = read_cell(src_reg, x, y, z);
                xpos_alive_count += val;
                write_cell(dst_arr_xpos, x_send, y, z, val);
            }
        }
        x_send++;
    }
    //printf("XPOS ALIVE: %lu, XNEG ALIVE: %lu\n", xpos_alive_count, xneg_alive_count);
}

int reg_size(region_t* reg){
    return reg->w * reg->h * reg->d;
}

void send_boundaries(MPI_Request* req1, MPI_Request* req2, int world_size, int my_rank, region_t* xneg_boundary, region_t* xpos_boundary){
    char errcode[MPI_MAX_ERROR_STRING];
    if(my_rank - 1 >= 0){
        int code = MPI_Issend(xneg_boundary->data, reg_size(xneg_boundary), MPI_UNSIGNED_CHAR, my_rank - 1, EDGE_LEFT_TAG_BASE + my_rank, MPI_COMM_WORLD, req1);
        if(code != MPI_SUCCESS){
            int cnt;
            MPI_Error_string(code, errcode, &cnt);
            printf("RANK %d MPI ERROR: %s\n", my_rank, errcode);
        }
    }
    else {
        *req1 = MPI_REQUEST_NULL;
    }

    if(my_rank + 1 < world_size){
        int code = MPI_Issend(xpos_boundary->data, reg_size(xpos_boundary), MPI_UNSIGNED_CHAR, my_rank + 1, EDGE_RIGHT_TAG_BASE + my_rank, MPI_COMM_WORLD, req2);
        if(code != MPI_SUCCESS){
            int cnt;
            MPI_Error_string(code, errcode, &cnt);
            printf("RANK %d MPI ERROR: %s\n", my_rank, errcode);
        }
    }
    else{
        *req2 = MPI_REQUEST_NULL;
    }
}

void await_sends(MPI_Request* req1, MPI_Request* req2){
    MPI_Status s;
    char errcode[MPI_MAX_ERROR_STRING];
    if(*req1 != MPI_REQUEST_NULL){
        int err = MPI_Wait(req1, &s);
        if(err != MPI_SUCCESS){
            printf("MPI Wait error: %d\n", err);
            int cnt;
            MPI_Error_string(err, errcode, &cnt);
            printf("MPI ERROR: %s\n", errcode);
        }
    }
    if(*req2 != MPI_REQUEST_NULL){
        int err = MPI_Wait(req2, &s);
        if(err != MPI_SUCCESS){
            printf("MPI Wait error: %d\n", err);
            int cnt;
            MPI_Error_string(err, errcode, &cnt);
            printf("MPI ERROR: %s\n", errcode);
        }
    }
}


void receive_boundaries(int world_size, int recv_from_xneg, int recv_from_xpos, region_t* xneg_dst_reg, region_t* xpos_dst_reg){
    if(recv_from_xneg >= 0){
        MPI_Status s;
        MPI_Recv(xneg_dst_reg->data, reg_size(xneg_dst_reg), MPI_UNSIGNED_CHAR, recv_from_xneg, EDGE_RIGHT_TAG_BASE + recv_from_xneg, MPI_COMM_WORLD, &s);
        if(s.MPI_ERROR != MPI_SUCCESS){
            printf("MPI RECV error: %d\n", s.MPI_ERROR);
        }
    }
    if(recv_from_xpos < world_size){
        MPI_Status s;
        MPI_Recv(xpos_dst_reg->data, reg_size(xpos_dst_reg), MPI_UNSIGNED_CHAR, recv_from_xpos, EDGE_LEFT_TAG_BASE + recv_from_xpos, MPI_COMM_WORLD, &s);
        if(s.MPI_ERROR != MPI_SUCCESS){
            printf("MPI RECV error: %d\n", s.MPI_ERROR);
        }
    }
}

void write_in_boundary_regions(region_t* dst_reg, region_t* incoming_xneg, region_t* incoming_xpos){
    uint64_t xpos_alive = 0;
    uint64_t xneg_alive = 0;
    for(int x = 0; x < incoming_xneg->w; x++){
        for(int y = 0; y < incoming_xneg->h; y++){
            for(int z = 0; z < incoming_xneg->d; z++){
                uint8_t val = read_cell(incoming_xneg, x, y, z);
                xneg_alive += val;
                write_cell(dst_reg, x, y, z, val);
                uint8_t val2 = read_cell(incoming_xpos, x, y, z);
                xpos_alive += val2;
                //region is (width + 2 * (incoming->w)) wide
                //want to start writing at (width + (incoming->w)) = regwidth - (incoming->w * 2) + (incoming->w)
                write_cell(dst_reg, x + (dst_reg->w - incoming_xpos->w), y, z, val2);
            }
        }
    }
    //printf("RECEIVED XPOS ALIVE: %lu, XNEG ALIVE: %lu\n", xpos_alive, xneg_alive);
}


uint64_t do_math(region_t* src_reg, region_t* dst_reg, int dist, uint64_t* alive){
    uint64_t sum = 0;
    uint64_t src_alive = 0;
    for(size_t x = dist; x < src_reg->w - dist; x++){
        for(size_t y = dist; y < src_reg->h - dist; y++){
            for(size_t z = dist; z < src_reg->d - dist; z++){
                sum += apply_rule(src_reg, dst_reg, x, y, z, dist);
                src_alive += read_cell(src_reg, x, y, z);
            }
        }
    }
    //printf("ALIVE: %lu\n", src_alive);
    *alive = src_alive;
    return sum;
}

typedef struct boundaries_s {
    region_t* xneg_send;
    region_t* xneg_recv;
    region_t* xpos_send;
    region_t* xpos_recv;
} boundaries_t;

uint64_t compute_step(region_t* src_reg, region_t* dst_reg, boundaries_t* bounds, int dist, int world_size, int my_rank, int last_step, uint64_t* alive){
    MPI_Request r0 = MPI_REQUEST_NULL;
    MPI_Request r1 = MPI_REQUEST_NULL;
    //printf("%d STARTING SENDS\n", my_rank);
    get_boundary_regions(src_reg, dist, bounds->xneg_send, bounds->xpos_send);
    send_boundaries(&r0, &r1, world_size, my_rank, bounds->xneg_send, bounds->xpos_send);
    //printf("%d SENT BOUNDARY\n", my_rank);
    receive_boundaries(world_size, my_rank - 1, my_rank + 1, bounds->xneg_recv, bounds->xpos_recv);
    //printf("%d RECV BOUNDARY\n", my_rank);
    await_sends(&r0, &r1);
    write_in_boundary_regions(src_reg, bounds->xneg_recv, bounds->xpos_recv);
    //printf("GOT GHOST CELLS\n");
    uint64_t num_updated = do_math(src_reg, dst_reg, dist, alive);
    if(last_step){
        return num_updated;
    }

    return num_updated;
}

void swap_regions(region_t** reg_1, region_t** reg_2){
    region_t* tmp_a = *reg_1;
    region_t* tmp_b = *reg_2;
    *reg_1 = tmp_b;
    *reg_2 = tmp_a;
}

void run_step_worker(region_t** src_reg, region_t** dst_reg, boundaries_t* bounds, int dist, int world_size, int my_rank, int last_step){
    uint64_t alive;
    uint64_t num_updated = compute_step(*src_reg, *dst_reg, bounds, dist, world_size, my_rank, last_step, &alive);
    uint64_t tmp[2] = {num_updated, alive};
    MPI_Send(&tmp, 2, MPI_UNSIGNED_LONG, 0, UPDATED_TAG + my_rank, MPI_COMM_WORLD);
    swap_regions(src_reg, dst_reg);
    MPI_Barrier(MPI_COMM_WORLD);

}

uint64_t collect_updated(int world_size, uint64_t* num_alive){
    uint64_t sum = 0;
    uint64_t alive = 0;
    MPI_Status s;
    for(int i = 1; i < world_size; i++){
        uint64_t tmp[2];
        MPI_Recv(&tmp, 2, MPI_UNSIGNED_LONG, i, UPDATED_TAG + i, MPI_COMM_WORLD, &s);
        //printf("REG %d: %lu updated\n", i, tmp);
        sum += tmp[0];
        alive += tmp[1];
    }
    *num_alive = alive;
    return sum;
}

uint64_t run_step_master(region_t** src_reg, region_t** dst_reg, boundaries_t* bounds, int dist, int world_size, int my_rank, int last_step){
    uint64_t alive;
    uint64_t num_updated = compute_step(*src_reg, *dst_reg, bounds, dist, world_size, my_rank, last_step, &alive);
    uint64_t other_alive;
    num_updated += collect_updated(world_size, &other_alive);
    printf("INPUT HAD %lu ALIVE\n", alive + other_alive);
    swap_regions(src_reg, dst_reg);
    MPI_Barrier(MPI_COMM_WORLD);
    return num_updated;
}

void populate_init_region(region_t* reg, int rank, int edge_len_no_dist, int actual_world_x, int x_len_no_dist, int dist){
    region_t* init_reg = init_region(actual_world_x, edge_len_no_dist, edge_len_no_dist);
    box_bounds_t bounds = {
        .x_min = X_MIN,
        .x_max = X_MAX,
        .y_min = Y_MIN,
        .y_max = Y_MAX,
        .z_min = Z_MIN,
        .z_max = Z_MAX,
    };

    fill_region(init_reg, &bounds);

    int x_offset = x_len_no_dist * rank;
    uint64_t live_cells = 0;
    for(int x = 0; x < x_len_no_dist; x++){
        for(int y = 0; y < edge_len_no_dist; y++){
            for(int z = 0; z < edge_len_no_dist; z++){
                uint8_t value = read_cell(init_reg, x + x_offset, y, z);
                write_cell(reg, dist + x, dist + y, dist + z, value);
                if(value){
                    live_cells++;
                }
            }
        }
    }

    free_region(init_reg);
    printf("rank %d x offset = %d, starting_live = %lu\n", rank, x_offset, live_cells);
}

int main(int argc, char** argv){
    int edge_len_no_dist = 0;
    int iters = 0;
    if(argc != 3){
        printf("Usage: <edge length> <num iterations>\n");
        return 1;
    }
    else {
        edge_len_no_dist = atoi(argv[1]);
        iters = atoi(argv[2]);
        //printf("Running %d iters with edge length: %d\n", iters, edge_len_no_dist);
    }

    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0){
        printf("Running %d iters with edge length: %d\n", iters, edge_len_no_dist);
    }
    
    int x_len = 0;
    int actual_world_x = 0;
    if(edge_len_no_dist % world_size != 0){
        actual_world_x = (edge_len_no_dist / world_size) * (world_size + 1);
        printf("Edge len %d not evenly divisible by world size %d, rounding x edge_len up to %d\n", edge_len_no_dist, world_size, x_len);
        x_len = actual_world_x / world_size;
    }
    else {
        actual_world_x = edge_len_no_dist;
        x_len = edge_len_no_dist / world_size;
    }

    int dist = DISTANCE;
    int x_len_dist = x_len + (2 * dist);
    int edge_len = edge_len_no_dist + (2 * dist);

    region_t* reg_a = init_region(x_len_dist, edge_len, edge_len);
    region_t* reg_b = init_region(x_len_dist, edge_len, edge_len);
    boundaries_t bounds = { 
        .xneg_send = init_region(dist, edge_len, edge_len),
        .xneg_recv = init_region(dist, edge_len, edge_len),
        .xpos_send = init_region(dist, edge_len, edge_len),
        .xpos_recv = init_region(dist, edge_len, edge_len)
    };

    populate_init_region(reg_a, rank, edge_len_no_dist, actual_world_x, x_len, dist);

    if(rank == 0){
        for(int step = 0; step < iters; step++){
            auto t0 = Time::now();
            printf("starting iter %d\n", step);
            int last_step = (step == iters - 1);
            uint64_t updated = run_step_master(&reg_a, &reg_b, &bounds, dist, world_size, rank, last_step);
            auto t1 = Time::now();
            fsec_t fs = t1 - t0;
            ms_t dur = std::chrono::duration_cast<ms_t>(fs);
            printf("iter %d: %lu cells updated in %dms\n", step, updated, dur.count());
        }
    }
    else {
        for(int step = 0; step < iters; step++){
            int last_step = (step == iters - 1);
            run_step_worker(&reg_a, &reg_b, &bounds, dist, world_size, rank, last_step);
        }
    }

    MPI_Finalize();
}


