#pragma once 
#include <inttypes.h>
#include <chrono>
#include <cstdlib>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms_t;
typedef std::chrono::duration<float> fsec_t;

//BOUNDARIES OF THE STARTING SHAPE
#define X_MIN 200
#define X_MAX 600
#define Y_MIN 350
#define Y_MAX 700
#define Z_MIN 500
#define Z_MAX 900

//RULES FOR UPDATING
#define MIN_SURVIVE 4
#define MAX_SURVIVE 15
#define MIN_BORN 8
//IN ORDER FOR DISTANCE TO WORK, YOU NEED TO DISABLE FAST3x3x3MOORE
//IF FAST3x3x3MOORE IS ENABLED, YOU MUST SET DISTANCE=1
#undef FAST3x3x3MOORE
//searches a (1 + 2*DISTANCE) x (1 + 2*DISTANCE) + (1 + 2*DISTANCE) cube around the cell being evaluated
#define DISTANCE 2 

typedef struct region_s {
    uint8_t* data;
    size_t w;
    size_t h;
    size_t d; 
} region_t;

region_t* init_region(size_t w, size_t h, size_t d){
    region_t* reg = (region_t*)malloc(sizeof(region_t));
    reg->w = w;
    reg->h = h;
    reg->d = d;
    reg->data = (uint8_t*)calloc(w * d * h, sizeof(uint8_t));
    //printf("new region: %lux%lux%lu\n", w, h, d);
    return reg; 
}

void free_region(region_t* reg){
    if(reg == NULL){
        return;
    }
    if(reg->data != NULL){
        free(reg->data);
    }
    free(reg);
}

inline uint8_t read_cell(region_t* reg, size_t x, size_t y, size_t z){
    //return reg->data[(z * reg->w * reg->h) + (y * reg->w) + x]; //takes almost twice as long in openmp
    return reg->data[(x * reg->d * reg->h) + (y * reg->d) + z];
}

inline void write_cell(region_t* reg, size_t x, size_t y, size_t z, uint8_t val){
    //reg->data[(z * reg->w * reg->h) + (y * reg->w) + x] = val; //takes almost twice as long in openmp
    reg->data[(x * reg->d * reg->h) + (y * reg->d) + z] = val;
}

typedef struct box_bounds_s {
    size_t x_min;
    size_t x_max;
    size_t y_min;
    size_t y_max;
    size_t z_min;
    size_t z_max;
} box_bounds_t;

void fill_region(region_t* reg, box_bounds_t* bb){
    for(size_t i = bb->x_min; i < bb->x_max; i++){
        for(size_t j = bb->y_min; j < bb->y_max; j++){
            for(size_t k = bb->z_min; k < bb->z_max; k++){
                write_cell(reg, i, j, k, 1);
            }
        }
    }
}

//fast version for a 3x3x3 cube, I think the compiler has an easier time unrolling the loops when it's like this
#ifdef FAST3x3x3MOORE
inline int moore_3d_v2(region_t* reg, size_t x, size_t y, size_t z){
    int sum = 0;
    for(int dx = x - 1; dx < x + 2; dx++){
        for(int dy = y - 1; dy < y + 2; dy++){
            for(int dz = z - 1; dz < z + 2; dz++){
                sum += read_cell(reg, dx, dy, dz);
            }
        }
    }
    return sum;
}
#else 
//slow version with a parameterized distance
inline int moore_3d_v2(region_t* reg, size_t x, size_t y, size_t z){
    int sum = 0;
    for(int dx = x - DISTANCE; dx < x + 1 + DISTANCE; dx++){
        for(int dy = y - DISTANCE; dy < y + 1 + DISTANCE; dy++){
            for(int dz = z - DISTANCE; dz < z + 1 + DISTANCE; dz++){
                sum += read_cell(reg, dx, dy, dz);
            }
        }
    }
    return sum;
}
#endif


//might be worth inlining this 
uint64_t apply_rule(region_t* src_reg, region_t* dst_reg, size_t x, size_t y, size_t z, int dist){
    int sum = moore_3d_v2(src_reg, x, y, z);
    uint8_t cur_val = read_cell(src_reg, x, y, z);
    if(cur_val == 0){
        if(sum >= MIN_BORN){
            write_cell(dst_reg, x, y, z, 1);
            return 1;
        }
        else {
            write_cell(dst_reg, x, y, z, 0);
        }
    }
    else{
        if(sum >= MIN_SURVIVE && sum <= MAX_SURVIVE){
           write_cell(dst_reg, x, y, z, 1);
        }
        else{
            write_cell(dst_reg, x, y, z, 0);
            return 1;
        }
    }
    return 0;
}
