file(GLOB proto_files *.proto)
protobuf_generate_cpp(proto_srcs proto_hdrs EXPORT_MACRO CAFFE_EXPORT ${proto_files})
protobuf_generate_python(proto_python ${proto_files})

SET_SOURCE_FILES_PROPERTIES(${proto_srcs} ${proto_hdrs} PROPERTIES GENERATED TRUE)
include_directories("${PROJECT_BINARY_DIR}/src")

add_custom_target(generate_proto ALL DEPENDS ${proto_srcs} ${proto_hdrs} ${proto_python})

set( proto_srcs ${proto_srcs} PARENT_SCOPE )
set( proto_hdrs ${proto_hdrs} PARENT_SCOPE )
set( proto_python ${proto_python} PARENT_SCOPE )