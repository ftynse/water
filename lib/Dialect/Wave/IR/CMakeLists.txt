add_mlir_dialect_library(MLIRWaveDialect
  WaveInterfaces.cpp
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

# Install the Wave dialect library so Python can find it at runtime.
install(TARGETS MLIRWaveDialect
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
