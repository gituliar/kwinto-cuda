cmake_minimum_required(VERSION 3.18.0)

if(DEFINED ENV{VCPKG_ROOT})
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    endif()
endif()

#set(VCPKG_FEATURE_FLAGS "versions")
#if(DEFINED WIN32)
#set(VCPKG_TARGET_TRIPLET x64-windows-static)
#endif()


string(TIMESTAMP KWINTO_VERSION "%Y-%m-%d %H:%M")
add_compile_definitions(KWINTO_VERSION="${KWINTO_VERSION}")

project(kwinto LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 20)

# delete 2 lines below since v3.25.2
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(src)
add_subdirectory(test)