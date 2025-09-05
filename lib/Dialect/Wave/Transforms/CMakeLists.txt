add_mlir_dialect_library(MLIRLowerWaveTransforms
  LowerRegister.cpp
  LowerWaveToMLIR.cpp
  TypeConverter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water/

  DEPENDS
  MLIRLowerWavePassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRWaveDialect
)
