cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0077 NEW)
project(falprs)
set(TARGET_NAME "${PROJECT_NAME}")
set(CMAKE_CXX_STANDARD 20)

option(BUILD_LPRS "Build with LPRS components." ON)
option(BUILD_FRS "Build with FRS components." ON)

if (NOT BUILD_LPRS AND NOT BUILD_FRS)
  message(FATAL_ERROR, "At least one of the options BUILD_LPRS or BUILD_FRS must be turned on")
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-gcc-compat -mavx2)

set(ABSL_PROPAGATE_CXX_STD ON)
set(TRITON_CLIENT_DIR "$ENV{HOME}/triton-client/build/install" CACHE STRING "Triton client directory")
set(CURL_ZLIB OFF)
set(USERVER_FEATURE_CORE ON)
set(USERVER_FEATURE_MONGODB OFF)
set(USERVER_FEATURE_POSTGRESQL ON)
set(USERVER_FEATURE_REDIS OFF)
set(USERVER_FEATURE_CLICKHOUSE OFF)
set(USERVER_FEATURE_GRPC OFF)
set(USERVER_FEATURE_KAFKA OFF)
set(USERVER_FEATURE_RABBITMQ OFF)
set(USERVER_FEATURE_MYSQL OFF)
set(USERVER_FEATURE_ROCKS OFF)
set(USERVER_FEATURE_YDB OFF)
set(USERVER_FEATURE_UTEST OFF)
set(USERVER_FEATURE_CRYPTOPP_BLAKE2 OFF)
set(USERVER_FEATURE_PATCH_LIBPQ ON)
set(USERVER_FEATURE_CRYPTOPP_BASE64_URL OFF)
set(USERVER_FEATURE_REDIS_HI_MALLOC OFF)
set(USERVER_FEATURE_REDIS_TLS OFF)
set(USERVER_FEATURE_STACKTRACE OFF)
set(USERVER_FEATURE_JEMALLOC ON)
set(USERVER_FEATURE_DWCAS ON)
set(USERVER_FEATURE_TESTSUITE ON)
set(USERVER_FEATURE_GRPC_CHANNELZ OFF)
set(USERVER_CHECK_PACKAGE_VERSIONS ON)
set(USERVER_LTO OFF)
set(USERVER_USE_CCACHE OFF)
set(USERVER_COMPILATION_TIME_TRACE OFF)
set(USERVER_NO_WERROR ON)
set(USERVER_FEATURE_ERASE_LOG_WITH_LEVEL OFF)
set(USERVER_DOWNLOAD_PACKAGES ON)
set(USERVER_PIP_USE_SYSTEM_PACKAGES OFF)
set(USERVER_IS_THE_ROOT_PROJECT OFF)
set(USERVER_PG_SERVER_INCLUDE_DIR "/usr/include/postgresql/14/server")
set(USERVER_PG_SERVER_LIBRARY_DIR "/usr/lib/postgresql/14/lib")
set(USERVER_PG_INCLUDE_DIR "/usr/include/postgresql")
set(USERVER_PG_LIBRARY_DIR "/usr/lib/postgresql/14/lib")
set(USERVER_DISABLE_PHDR_CACHE ON)

find_package(OpenCV REQUIRED)
list(APPEND TRITON_CLIENT_INCLUDE_DIRS ${TRITON_CLIENT_DIR}/include)
list(APPEND TRITON_CLIENT_LIBS ${TRITON_CLIENT_DIR}/lib/libhttpclient_static.a)

add_subdirectory(contrib/userver)
add_subdirectory(contrib/abseil-cpp)

list(APPEND SOURCES main.cpp)
if (BUILD_LPRS)
  add_definitions(-DBUILD_LPRS)
  list(APPEND SOURCES
    lprs_caches.hpp
    lprs_api.hpp
    lprs_api.cpp
    lprs_workflow.hpp
    lprs_workflow.cpp)
endif()
if (BUILD_FRS)
  add_definitions(-DBUILD_FRS)
  list(APPEND SOURCES
    frs_caches.hpp
    frs_api.hpp
    frs_api.cpp
    frs_workflow.hpp
    frs_workflow.cpp)
endif()
add_executable(${TARGET_NAME} ${SOURCES})

# include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${TRITON_CLIENT_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS} ${TRITON_CLIENT_LIBS} userver::core userver::postgresql absl::strings absl::str_format absl::time absl::flat_hash_map absl::flat_hash_set dl)
