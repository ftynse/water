add_mlir_dialect_library(MLIRWaterTransforms
  AccessCheckers.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water

  DEPENDS
  MLIRWaterPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRVectorDialect
)
