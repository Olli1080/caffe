# These lists are later turned into target properties on main caffe library target
set(Caffe_LINKER_LIBS "")
set(Caffe_INCLUDE_DIRS "")
set(Caffe_DEFINITIONS "")
set(Caffe_COMPILE_OPTIONS "")

# ---[ Boost
find_package(Boost 1.80 REQUIRED COMPONENTS system thread filesystem)
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${Boost_LIBRARIES})

# ---[ Threads
#find_package(Threads REQUIRED)
#list(APPEND Caffe_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ OpenMP
if(USE_OPENMP)
  # Ideally, this should be provided by the BLAS library IMPORTED target. However,
  # nobody does this, so we need to link to OpenMP explicitly and have the maintainer
  # to flick the switch manually as needed.
  #
  # Moreover, OpenMP package does not provide an IMPORTED target as well, and the
  # suggested way of linking to OpenMP is to append to CMAKE_{C,CXX}_FLAGS.
  # However, this naïve method will force any user of Caffe to add the same kludge
  # into their buildsystem again, so we put these options into per-target PUBLIC
  # compile options and link flags, so that they will be exported properly.
  find_package(OpenMP REQUIRED)
  list(APPEND Caffe_LINKER_LIBS PRIVATE ${OpenMP_CXX_FLAGS})
  list(APPEND Caffe_COMPILE_OPTIONS PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ---[ Google-glog
find_package(glog CONFIG REQUIRED)
#include("cmake/External/glog.cmake")
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC glog::glog)

# ---[ Google-gflags
find_package(gflags CONFIG REQUIRED)
#include("cmake/External/gflags.cmake")
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC gflags::gflags)

# ---[ Google-protobuf
find_package(Protobuf REQUIRED)
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Protobuf_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${Protobuf_LIBRARIES})

# ---[ HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${HDF5_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# This code is taken from https://github.com/sh1r0/caffe-android-lib
if(USE_HDF5)
  find_package(HDF5 COMPONENTS HL REQUIRED)
  include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
  add_definitions(-DUSE_HDF5)
endif()

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${LMDB_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${LMDB_LIBRARIES})
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LMDB)
  if(ALLOW_LMDB_NOLOCK)
    list(APPEND Caffe_DEFINITIONS PRIVATE -DALLOW_LMDB_NOLOCK)
  endif()
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${LevelDB_INCLUDES})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${LevelDB_LIBRARIES})
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LEVELDB)
endif()

# ---[ Snappy
if(USE_LEVELDB)
  find_package(Snappy REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${Snappy_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PRIVATE ${Snappy_LIBRARIES})
endif()

# ---[ CUDA
#include(cmake/Cuda.cmake)
if (CPU_ONLY)
  message(STATUS "-- CUDA is disabled. Building without it...")
  list(APPEND Caffe_DEFINITIONS PUBLIC -DCPU_ONLY)
else()
  find_package(CUDAToolkit)
  find_package(CUDNN)
  
  if(NOT CUDAToolkit_FOUND OR NOT CUDNN_FOUND)
    message(FATAL_ERROR "-- CUDA and/or cudnn are/is not detected by cmake. But was enabled. ERROR")
  endif()
  
  if (BUILD_SHARED_LIBS)
    list(APPEND Caffe_LINKER_LIBS CUDA::cudart CUDA::cublas CUDA::curand)
  else()
    list(APPEND Caffe_LINKER_LIBS CUDA::cudart_static CUDA::cublas_static CUDA::curand_static)
  endif()
  list(APPEND Caffe_LINKER_LIBS CuDNN::CuDNN)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${CUDNN_INCLUDE_DIRS})
endif()

if(USE_NCCL)
  find_package(NCCL REQUIRED)
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${NCCL_LIBRARIES})
  add_definitions(-DUSE_NCCL)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_OPENCV)
endif()

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Atlas_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenBLAS_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenBLAS_LIB})
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${MKL_LIBRARIES})
    list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${vecLib_LINKER_LIBS})

  if(VECLIB_FOUND)
    if(NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
      list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_ACCELERATE)
    endif()
  endif()
endif()

# ---[ Python
if(BUILD_python)
  if(NOT "${python_version}" VERSION_LESS "3.0.0")
    # use python3
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m ensurepip)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install numpy --no-warn-script-location)

    find_package(Python3 REQUIRED COMPONENTS NumPy)
    find_package(Boost 1.80 REQUIRED COMPONENTS python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})

    if(BUILD_python_layer)
      list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
      list(APPEND Caffe_LINKER_LIBS PRIVATE ${Python3_LIBRARIES})
    endif()
  else()
    # use python2
    find_package(Python2 REQUIRED COMPONENTS Interpreter Development)
    execute_process(COMMAND ${Python2_EXECUTABLE} -m ensurepip)
    execute_process(COMMAND ${Python2_EXECUTABLE} -m pip install numpy --no-warn-script-location)

    find_package(Python2 REQUIRED COMPONENTS NumPy)
    find_package(Boost 1.80 REQUIRED COMPONENTS python${Python2_VERSION_MAJOR}${Python2_VERSION_MINOR})

    if(BUILD_python_layer)
      list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
      list(APPEND Caffe_LINKER_LIBS PRIVATE ${Python2_LIBRARIES})
    endif()
  endif()
  if(BUILD_python_layer)
    list(APPEND Caffe_DEFINITIONS PRIVATE -DWITH_PYTHON_LAYER)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${Boost_LIBRARIES})
    #maybe only add Boost_<COMPONENT>_LIBRARY
  endif()
endif()

# ---[ Matlab
if(BUILD_matlab)
  find_package(MatlabMex)
  if(MATLABMEX_FOUND)
    set(HAVE_MATLAB TRUE)
  endif()

  # sudo apt-get install liboctave-dev
  find_program(Octave_compiler NAMES mkoctfile DOC "Octave C++ compiler")

  if(HAVE_MATLAB AND Octave_compiler)
    set(Matlab_build_mex_using "Matlab" CACHE STRING "Select Matlab or Octave if both detected")
    set_property(CACHE Matlab_build_mex_using PROPERTY STRINGS "Matlab;Octave")
  endif()
endif()

# ---[ Doxygen
if(BUILD_docs)
  find_package(Doxygen)
endif()
