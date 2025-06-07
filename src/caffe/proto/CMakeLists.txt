file(GLOB proto_files *.proto)
protobuf_generate_cpp(proto_srcs proto_hdrs EXPORT_MACRO CAFFE_EXPORT ${proto_files})
if(BUILD_python)
    protobuf_generate_python(proto_python ${proto_files})
endif()

SET_SOURCE_FILES_PROPERTIES(${proto_srcs} ${proto_hdrs} ${proto_python} PROPERTIES GENERATED TRUE)

add_custom_target(generate_proto DEPENDS ${proto_srcs} ${proto_hdrs} ${proto_python})
add_custom_command(TARGET generate_proto POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/include/caffe/proto/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${proto_hdrs} "${PROJECT_BINARY_DIR}/include/caffe/proto/"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${Caffe_INCLUDE_DIR}/caffe/proto/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${proto_hdrs} "${Caffe_INCLUDE_DIR}/caffe/proto/"
    COMMENT "Copied Generated Files"
)

foreach(src IN LISTS proto_hdrs)
    get_filename_component(file ${src} NAME)
    #message("src=${file}")
    SET_SOURCE_FILES_PROPERTIES(
        "${PROJECT_BINARY_DIR}/include/caffe/proto/${file}" 
        "${Caffe_INCLUDE_DIR}/caffe/proto/${file}"
        PROPERTIES GENERATED TRUE)
endforeach()

#SET_SOURCE_FILES_PROPERTIES(${proto_srcs} ${proto_hdrs} ${proto_python} PROPERTIES GENERATED TRUE)

set( proto_srcs ${proto_srcs} PARENT_SCOPE )
set( proto_hdrs ${proto_hdrs} PARENT_SCOPE )
set( proto_python ${proto_python} PARENT_SCOPE )