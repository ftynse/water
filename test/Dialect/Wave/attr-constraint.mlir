// RUN: water-opt %s | FileCheck %s

// CHECK: #wave.hardware_constraint<threads_per_wave = 64>
func.func private @test_hw1() attributes { wave.constraints = [#wave.hardware_constraint<threads_per_wave = 64>] }


// CHECK: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = <f32_16x16x16_f16>, vector_shapes = {K = 64 : i64, M = 1 : i64, N = 1 : i64}>
#hw_constraint2 = #wave.hardware_constraint<threads_per_wave = 64,
                                             waves_per_block = [1, 1, 1],
                                             mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                             vector_shapes = {M = 1, N = 1, K = 64},
                                             max_bits_per_load = 128>
func.func private @test_hw2() attributes { wave.constraints = [#hw_constraint2] }


// CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = 0>
#wg_constraint1 = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = 0>
func.func private @test_wg1() attributes { wave.constraints = [#wg_constraint1] }

// CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M]
// CHECK: -> (BLOCK_M floordiv 4), workgroup_dim = 0, iters = [M] -> (M), per_device_dim = <"DEVICE_M">>
#wg_constraint2 = #wave.workgroup_constraint<dim = <"M">,
                                             tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4),
                                             workgroup_dim = 0,
                                             primary = true,
                                             iters = [M] -> (M),
                                             per_device_dim = <"DEVICE_M">>
func.func private @test_wg2() attributes { wave.constraints = [#wg_constraint2] }


// CHECK: #wave.tiling_constraint<dim = <"M">>
func.func private @test_tiling1() attributes { wave.constraints = [#wave.tiling_constraint<dim = <"M">>] }


// CHECK: #wave.tiling_constraint<dim = <"K">, tile_size = [BLOCK_K] -> (BLOCK_K),
// CHECK:   induction_var = [ARGK] -> (ARGK), iters = [SPLIT_LEN, BLOCK_K] -> (SPLIT_LEN floordiv BLOCK_K), start = [SPLIT_OFF] -> (SPLIT_OFF)>
#tl_constraint2 = #wave.tiling_constraint<dim = <"K">,
                                          tile_size = [BLOCK_K] -> (BLOCK_K),
                                          induction_var = [ARGK] -> (ARGK),
                                          iters = [SPLIT_LEN, BLOCK_K] -> (SPLIT_LEN floordiv BLOCK_K),
                                          start = [SPLIT_OFF] -> (SPLIT_OFF)>
func.func private @test_tiling2() attributes { wave.constraints = [#tl_constraint2] }


// CHECK: #wave.wave_constraint<dim = <"K">, tile_size = [BLOCK_K] -> (BLOCK_K floordiv 4)>
#wv_constraint1 = #wave.wave_constraint<dim = <"K">, tile_size = [BLOCK_K] -> (BLOCK_K floordiv 4)>
func.func private @test_wave1() attributes { wave.constraints = [#wv_constraint1] }

// CHECK: #wave.wave_constraint<dim = <"K">, tile_size = [BLOCK_K] -> (BLOCK_K floordiv 4), wave_id = [THREAD_0] -> (THREAD_0 floordiv 64),
// CHECK:   wg_constraint = <dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = 0>
#wv_constraint2 = #wave.wave_constraint<dim = <"K">,
                                        tile_size = [BLOCK_K] -> (BLOCK_K floordiv 4),
                                        wave_id = [THREAD_0] -> (THREAD_0 floordiv 64),
                                        wg_constraint = #wg_constraint1>
func.func private @test_wave2() attributes { wave.constraints = [#wv_constraint2] }

// CHECK: #wave.reordering_constraint<reordered_equation = [WG0, WG1, M, BLOCK_M, GROUP_SIZE_N]
// CHECK:   -> (((WG1 * (M ceildiv BLOCK_M) + WG0) mod (GROUP_SIZE_N * (M ceildiv BLOCK_M))) floordiv GROUP_SIZE_N), workgroup_dim = 0>
#ro_constraint = #wave.reordering_constraint<reordered_equation = [WG0, WG1, M, BLOCK_M, GROUP_SIZE_N] ->
                                             ((((WG1 * (M ceildiv BLOCK_M) + WG0) mod (GROUP_SIZE_N * (M ceildiv BLOCK_M))) floordiv GROUP_SIZE_N)),
                                             workgroup_dim = 0>
func.func private @test_reodering() attributes { wave.constraints = [#ro_constraint] }


// CHECK: #wave.iterator_binding<{k = #wave.symbol<"K">, m = #wave.symbol<"M">, n = #wave.symbol<"N">}>
#ib_constraint = #wave.iterator_binding<{ m = #wave.symbol<"M">, n = #wave.symbol<"N">, k = #wave.symbol<"K"> }>
func.func private @test_iterator_binding() attributes { wave.constraints = [#ib_constraint] }


// CHECK: #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
#dv_constraint = #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
func.func private @test_device() attributes { wave.constraints = [#dv_constraint] }
