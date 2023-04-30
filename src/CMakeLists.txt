include(FetchContent)

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

  "kwAsset.cpp"
  "kwBenchmark.cpp"
  "kwCommon.cpp"
  "kwFd1d.cpp"
  "kwPortfolio.cpp"
  "kwString.cpp"

  "kwVersion.cpp"
)

# Link Docopt
if(DEFINED WIN32)
    # Use boost::regex due to CRASH on Win1x. See https://github.com/docopt/docopt.cpp/issues/67
    set(USE_BOOST_REGEX ON CACHE INTERNAL "docopt: Replace std::regex with Boost.Regex")
endif()
FetchContent_Declare(
  docopt
  GIT_SHALLOW    TRUE
  GIT_REPOSITORY https://github.com/docopt/docopt.cpp.git
  GIT_TAG        v0.6.3
)
FetchContent_MakeAvailable(docopt)


target_link_libraries(
    kwinto_lib
    PUBLIC
        kwinto_gpu

        CUDA::cusparse
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

        docopt_s
        #$<IF:$<TARGET_EXISTS:docopt>,docopt,docopt_s>
)
