add_mlir_dialect_library(MLIRWaveTransforms
  InferTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  DEPENDS
  MLIRWaterWavePassesIncGen

  LINK_LIBS PUBLIC
  MLIRWaveDialect
  MLIRPass
)
