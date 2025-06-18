add_mlir_dialect_library(MLIRWaterTransforms
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  DEPENDS
  MLIRWaterPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
