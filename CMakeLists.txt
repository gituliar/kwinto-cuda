cmake_minimum_required(VERSION 3.20.0)

if(DEFINED WIN32)
    set(VCPKG_TARGET_TRIPLET x64-windows-static)

    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    set(Boost_NO_WARN_NEW_VERSIONS ON)
endif()

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
