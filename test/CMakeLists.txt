# Test Section

add_executable(
    kwinto_test

    kwinto_test.cpp

    kwBlackScholes_test.cpp
    kwFd1d_test.cpp
    kwPortfolio_test.cpp
)

find_package(GTest REQUIRED)
enable_testing()

target_link_libraries(
    kwinto_test
    PRIVATE
        kwinto_lib

        GTest::gmock
        GTest::gtest
        GTest::gmock_main
        GTest::gtest_main
)

gtest_discover_tests(
    kwinto_test
)