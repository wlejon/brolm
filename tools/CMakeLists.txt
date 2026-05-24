set(STB_DIR "${CMAKE_SOURCE_DIR}/../bro/third_party/stb" CACHE PATH
    "Path to stb single-header library (needs stb_image.h)")
if(NOT EXISTS "${STB_DIR}/stb_image.h")
    message(WARNING
        "tools/: stb_image.h not found at ${STB_DIR} — run_qwen35_image will not be built.")
    return()
endif()

add_executable(brolm_run_qwen35_image run_qwen35_image.cpp)
target_link_libraries(brolm_run_qwen35_image PRIVATE brolm)
target_include_directories(brolm_run_qwen35_image PRIVATE "${STB_DIR}")
if(MSVC)
    target_compile_options(brolm_run_qwen35_image PRIVATE /W4)
else()
    target_compile_options(brolm_run_qwen35_image PRIVATE -Wall -Wextra)
endif()
