# generate protobuf sources
add_subdirectory(proto)
#file(GLOB proto_files *.proto)
#protobuf_generate_cpp(proto_srcs proto_hdrs EXPORT_MACRO CAFFE_EXPORT ${proto_files})
#protobuf_generate_python(proto_python ${proto_files})

#SET_SOURCE_FILES_PROPERTIES(${proto_srcs} ${proto_hdrs} PROPERTIES GENERATED TRUE)
#include_directories("${PROJECT_BINARY_DIR}/src")

#foreach(genFile ${proto_srcs} ${proto_hdrs})
#  cmake_path(GET genFile FILENAME fname)
#  set(fname "${PROJECT_BINARY_DIR}/include/caffe/proto/${fname}")
#  file(GENERATE OUTPUT "${fname}"
#    INPUT "${genFile}"
#  )
#  SET_SOURCE_FILES_PROPERTIES(${fname} PROPERTIES GENERATED TRUE)
#endforeach()     

# collect any compile definitions from imported targets. This important so that
# preprocessor macros such as GLOG_NO_ABBREVIATED_SEVERITIES are defined.
# this is required since CUDA macros do not honor the INTERFACE_COMPILE_DEFINITIONS
caffe_collect_compile_definitions(Caffe_COMPILE_DEFS
                                  ${Caffe_LINKER_LIBS})

# set the MSVC export macros
if(BUILD_SHARED_LIBS AND MSVC)
  set(Caffe_EXPORT_DEFINITION -DCAFFE_EXPORT=__declspec\(dllexport\))
  set(Caffe_IMPORT_DEFINITION -DCAFFE_EXPORT=__declspec\(dllimport\))
else()
  set(Caffe_EXPORT_DEFINITION -DCAFFE_EXPORT=)
  set(Caffe_IMPORT_DEFINITION -DCAFFE_EXPORT=)
endif()
set(Caffe_EXPORT_DEFINITIONS PRIVATE ${Caffe_EXPORT_DEFINITION}
                             INTERFACE ${Caffe_IMPORT_DEFINITION})

# include python files either to force generation
# make caffeproto an object library so the symbols get exported on windows
add_library(caffeproto OBJECT ${proto_hdrs} ${proto_srcs} ${proto_python})
caffe_default_properties(caffeproto)
target_include_directories(caffeproto PUBLIC ${Protobuf_INCLUDE_DIRS})
target_compile_definitions(caffeproto ${Caffe_EXPORT_DEFINITIONS}
                                      PRIVATE ${Caffe_COMPILE_DEFS})
target_link_libraries(caffeproto ${PROTOBUF_LIBRARY})
add_dependencies(caffeproto generate_proto)

# --[ Caffe library
# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(CUDAToolkit_FOUND)
  list(APPEND srcs ${cuda})
endif()
if(MSVC)
  list(APPEND Caffe_DEFINITIONS PUBLIC -DNO_STRICT)
endif()
add_library(caffe ${srcs} $<TARGET_OBJECTS:caffeproto>)
caffe_default_properties(caffe)
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
target_include_directories(caffe ${Caffe_INCLUDE_DIRS}
                                 PUBLIC
                                 $<BUILD_INTERFACE:${Caffe_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>
                                 )
target_compile_definitions(caffe ${Caffe_DEFINITIONS} ${Caffe_EXPORT_DEFINITIONS})
if(Caffe_COMPILE_OPTIONS)
  target_compile_options(caffe ${Caffe_COMPILE_OPTIONS})
endif()
set_target_properties(caffe PROPERTIES
    VERSION   ${CAFFE_TARGET_VERSION}
    SOVERSION ${CAFFE_TARGET_SOVERSION}
    )
if(MSVC AND USE_NCCL)
  add_dependencies(caffe nccl_external)
endif()

include(GenerateExportHeader)
generate_export_header(caffe
                       EXPORT_MACRO_NAME CAFFE_EXPORT
                       EXPORT_FILE_NAME ${caffe_export_hdr})

# ---[ Tests
add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${proto_hdrs} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/caffe/proto)
install(TARGETS caffe EXPORT CaffeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(MSVC)
  set(caffe_additional_hdr ${caffe_export_hdr})
  install(FILES ${caffe_additional_hdr} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/caffe)
endif()

if(BUILD_python)
  file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
  list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
  install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)
endif()
  
