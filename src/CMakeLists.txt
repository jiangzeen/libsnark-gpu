include_directories(.)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(main snark gpuec)
target_include_directories(
  main

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
  ${DEPENDS_DIR}/libsnark/depends/libff
  /usr/local/cuda/include
)