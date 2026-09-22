add_library(brolm_api STATIC
    api.cpp
    host_class.cpp
    native_lm_tokenizer.cpp
    native_lm_model.cpp
    native_lm_vl.cpp
    native_lm_clip.cpp
    native_lm_laya.cpp
)

target_include_directories(brolm_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(brolm_api PUBLIC
    brolm
    brotensor::brotensor
    bronze_runtime_shared
)

find_package(Threads REQUIRED)
target_link_libraries(brolm_api PUBLIC Threads::Threads)

target_compile_features(brolm_api PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brolm_api PRIVATE /MP /permissive- /utf-8 /W4)
else()
    target_compile_options(brolm_api PRIVATE -Wall -Wextra)
endif()
