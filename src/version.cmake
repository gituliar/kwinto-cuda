# Some practices to consider:
#   - https://semver.org/
#   - https://blog.aloni.org/posts/proper-use-of-git-tags/
#

execute_process(COMMAND git log --pretty=format:'%h' -n 1
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "N/A")
    set(GIT_DIFF "")
    set(GIT_TAG "N/A")
    set(GIT_BRANCH "N/A")
    set(GIT_DESCRIBE "N/A")
else()
    execute_process(
        COMMAND bash -c "git diff --quiet --exit-code || echo +"
        OUTPUT_VARIABLE GIT_DIFF)
    execute_process(
        COMMAND git describe --exact-match --tags
        OUTPUT_VARIABLE GIT_TAG ERROR_QUIET)
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        OUTPUT_VARIABLE GIT_BRANCH)
    execute_process(
        COMMAND git log -1 --format=%ci
        OUTPUT_VARIABLE GIT_DATE)
    execute_process(
        COMMAND git describe
        OUTPUT_VARIABLE GIT_DESCRIBE)

    string(STRIP "${GIT_REV}" GIT_REV)
    string(SUBSTRING "${GIT_REV}" 1 7 GIT_REV)
    string(STRIP "${GIT_DIFF}" GIT_DIFF)
    string(STRIP "${GIT_TAG}" GIT_TAG)
    string(STRIP "${GIT_BRANCH}" GIT_BRANCH)
    string(STRIP "${GIT_DATE}" GIT_DATE)
    string(STRIP "${GIT_DESCRIBE}" GIT_DESCRIBE)
endif()

execute_process(
    COMMAND bash -c "date +\"%Y-%m-%d %H:%M:%S %z\""
    OUTPUT_VARIABLE BUILD_DATE)
string(STRIP "${BUILD_DATE}" BUILD_DATE)

set(VERSION "#include \"kwVersion.h\"
const std::string kw::Version::BUILD_DATE=\"${BUILD_DATE}\";
const std::string kw::Version::GIT_BRANCH=\"${GIT_BRANCH}\";
const std::string kw::Version::GIT_DATE=\"${GIT_DATE}\";
const std::string kw::Version::GIT_DESCRIBE=\"${GIT_DESCRIBE}${GIT_DIFF}\";
const std::string kw::Version::GIT_REV=\"${GIT_REV}${GIT_DIFF}\";
const std::string kw::Version::GIT_TAG=\"${GIT_TAG}\";
")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kwVersion.cpp)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/kwVersion.cpp VERSION_)
else()
    set(VERSION_ "")
endif()

if (NOT "${VERSION}" STREQUAL "${VERSION_}")
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/kwVersion.cpp "${VERSION}")
endif()