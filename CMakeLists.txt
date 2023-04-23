cmake_minimum_required(VERSION 3.18.0)

if(DEFINED ENV{VCPKG_ROOT})
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    endif()
endif()


string(TIMESTAMP KWINTO_VERSION "%Y-%m-%d %H:%M")
add_compile_definitions(KWINTO_VERSION="${KWINTO_VERSION}")

project(kwinto LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)
add_subdirectory(test)
