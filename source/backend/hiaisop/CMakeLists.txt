file(GLOB_RECURSE MNN_HIAI_SOP_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_library(MNN_HiAISop OBJECT ${MNN_HIAI_SOP_SRCS})

target_include_directories(MNN_HiAISop PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(MNN_HiAISop PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backend/)
target_include_directories(MNN_HiAISop PRIVATE ${CMAKE_CURRENT_LIST_DIR}/execution/)
target_include_directories(MNN_HiAISop PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/)