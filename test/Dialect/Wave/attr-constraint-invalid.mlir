// RUN: water-opt %s --allow-unregistered-dialect --water-test-wave-dialect-constructors --split-input-file --verify-diagnostics

// expected-error @below {{waves_per_block (1) does should have the same size as vector_shapes ({M = 1 : i64, N = 64 : i64})}}
#hw_constraint = #wave.hardware_constraint<threads_per_wave = 64,
                                           waves_per_block = [1],
                                           mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                           vector_shapes = {M = 1, N = 64}>
func.func private @test_num_dimensions_mismatch1() attributes { wave.constraints = [#hw_constraint] }

// -----

// expected-error @below {{"M" is not an IntegerAttr: "BLOCK_M"}}
#hw_constraint = #wave.hardware_constraint<threads_per_wave = 64,
                                           waves_per_block = [1, 1],
                                           mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                           vector_shapes = {M = "BLOCK_M", N = 64}>
func.func private @test_num_dimensions_mismatch2() attributes { wave.constraints = [#hw_constraint] }

// -----

// expected-error @below {{"m" is not a WaveSymbolAttr: "M"}}
func.func private @invalid_iterator_binding1() attributes { wave.constraints = [#wave.iterator_binding<{ m = "M", n = "N" }>] }

// -----

// expected-error @below {{"m" is not a WaveSymbolAttr: 64}}
func.func private @invalid_iterator_binding2() attributes { wave.constraints = [#wave.iterator_binding<{ m = 64, n = 64 }>] }
