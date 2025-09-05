// RUN: water-opt %s | FileCheck %s

#hw_constraint1 = #wave.hardware_constraint<threads_per_wave = 64>

#hw_constraint2 = #wave.hardware_constraint<threads_per_wave = 64,
                                             waves_per_block = [1, 1, 1],
                                             mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                             vector_shapes = {M = 1, N = 1, K = 64},
                                             max_bits_per_load = 128>


#wg_constraint1 = #wave.workgroup_constraint<dim = [M] -> (M), tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = 0>

#wg_constraint2 = #wave.workgroup_constraint<dim = [M] -> (M),
                                             tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4),
                                             workgroup_dim = 0,
                                             primary = true,
                                             iters = [M] -> (M),
                                             per_device_dim = [DEVICE_M] -> (DEVICE_M)>


#tl_constraint1 = #wave.tiling_constraint<dim = [K] -> (K)>

#tl_constraint2 = #wave.tiling_constraint<dim = [K] -> (K),
                                          tile_size = [BLOCK_K] -> (BLOCK_K),
                                          induction_var = [ARGK] -> (ARGK),
                                          iters = [SPLIT_LEN, BLOCK_K] -> (SPLIT_LEN floordiv BLOCK_K),
                                          start = [SPLIT_OFF] -> (SPLIT_OFF)>


#wv_constraint1 = #wave.wave_constraint<dim = [K] -> (K), tile_size = [BLOCK_K] -> (BLOCK_K floordiv 4)>

#wv_constraint2 = #wave.wave_constraint<dim = [K] -> (K),
                                        tile_size = [BLOCK_K] -> (BLOCK_K floordiv 4),
                                        wave_id = [THREAD_0] -> (THREAD_0 floordiv 64),
                                        wg_constraint = #wg_constraint1>


#ro_constraint1 = #wave.reordering_constraint<reordered_equation = [WG0, WG1, M, BLOCK_M, GROUP_SIZE_N] -> ((((WG1 * (M ceildiv BLOCK_M) + WG0) mod (GROUP_SIZE_N * (M ceildiv BLOCK_M))) floordiv GROUP_SIZE_N)),
                                              workgroup_dim = 0>


#symM =#wave.symbol<"M">
#symN =#wave.symbol<"N">
#symK =#wave.symbol<"K">

#ib_constraint = #wave.iterator_binding<{ m = #symM, n = #symN, k = #symK }>


#dv_constraint1 = #wave.device_constraint<dim = [M] -> (M), tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
