### kwinto_gpu
add_library(
  kwinto_gpu

  "kwFd1d.cu"
)

# Link CUDA
find_package(CUDAToolkit REQUIRED)

set_property(TARGET kwinto_gpu PROPERTY CUDA_ARCHITECTURES OFF)
set_property(TARGET kwinto_gpu PROPERTY CUDA_STANDARD 20)

target_include_directories(
    kwinto_gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    kwinto_gpu

    PUBLIC
        CUDA::cudart
)

#target_compile_options(
#    kwinto_gpu
#    PRIVATE
#        $<$<COMPILE_LANGUAGE:CUDA>: -G>
#)


### kinto_lib

add_library(
  kwinto_lib

  "kwBenchmark.cpp"
  "kwBlackScholes.cpp"
  "kwCommon.cpp"
  "kwFd1d.cpp"

  "kwVersion.cpp"
)

# Link Docopt
find_package(
    docopt CONFIG REQUIRED)

target_link_libraries(
    kwinto_lib
    PUBLIC
        kwinto_gpu

        CUDA::cusparse
)

set_property(
    TARGET kwinto_lib PROPERTY
        POSITION_INDEPENDENT_CODE ON
)


### kwinto
add_executable(
  kwinto

  kwinto.cpp
)

# From https://www.mattkeeter.com/blog/2018-01-06-versioning/
include(version.cmake)

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/kwVersion.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/_kwVersion.cpp
    COMMAND
        ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake
)

target_link_libraries(
    kwinto
	PRIVATE
        kwinto_lib

        $<IF:$<TARGET_EXISTS:docopt>,docopt,docopt_s>
)
