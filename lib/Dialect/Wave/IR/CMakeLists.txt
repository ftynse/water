add_mlir_dialect_library(MLIRWaveDialect
  WaveAttrs.cpp
  WaveDialect.cpp
  WaveInterfaces.cpp
  WaveOps.cpp
  WaveTypes.cpp

  DEPENDS
  MLIRWaveIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
)
