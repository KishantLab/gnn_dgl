// #ifndef PART_ARRAY_CUH_
// #define PART_ARRAY_CUH_
//
// #include<stdio.h>
// #include <cstddef>
// #include <cstdint>
//
// int64_t* d_part_array;
// int  flag=0; 
//
// void initilize_global_array(int64_t *parts_array, size_t size)
// {
//   if(flag == 0)
//   {
//     // printf("flag activated");
//     flag = 1;
//     // allocate gpu memory
//     // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
//     cudaMalloc(&d_part_array, size * sizeof(int64_t));
//
//     cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
//     // printf("Cudamemcpy called");
//     // flag = 1;
//   }
//
//   // return d_part_array;
// }
// #endif //PART_ARRAY_CUH_

// global_array.h
#ifndef GLOBAL_ARRAY_H
#define GLOBAL_ARRAY_H

// Declare the global array (with extern to allow sharing between files)
#include <cstdint>
extern int64_t *d_part_array;  // 'extern' to share it across files.

// Declare the function prototypes
// void function1();  // Declares the function that initializes the array.
// COOMatrix _CSRRowWiseSamplingUniform1(
  // CSRMatrix mat, IdArray rows, const int64_t num_picks, NDArray part_array, const bool replace); 
#endif
