// RUN: water-opt %s | FileCheck %s

#wave.hardware_constraint<threads_per_wave = 64,
						  waves_per_block = [1, 1, 1],
						  mma_type = #TODO,
						  vector_shapes = {@B: 1, @M: 1, @N: 64},
						  max_bits_per_load = 128>

#wave.workgroup_constraint<dim = @M,
                           tile_size = @BLOCK_M floordiv 4,
                           workgroup_dim = 0,
                           primary = true,
                           iters = @M - 1,
                           per_device_dim = @M>

#wave.tiling_constraint<dim = @M,
                        tile_size = @BLOCK_M floordiv 4,
                        induction_var = @IDX,
                        iters = @SPLIT_LEN / @TILE_M,
                        start = 0>

#wave.wave_constraint<dim = @M,
                      tile_size = @BLOCK_M floordiv 4,
                      wave_id = @WAVE_M,
                      wg_constraint = #wg>

#wave.reordering_constraint<reordered_equation = @M, workgroup_dim = 0>


#wave.iterator_binding<{"m": @M, "k": @K, "nf": @NF}>


#wave.reordering_constraint<dim = @M,
                            tile_size = @DEVICE_M,
                            device_dim = 0>


func.func private @attr()
