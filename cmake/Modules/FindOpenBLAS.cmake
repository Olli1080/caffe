include(FindPackageHandleStandardArgs)

set(OPENBLAS_ROOT_DIR "" CACHE PATH "Folder contains OpenBlas")

find_path(OpenBLAS_INCLUDE_DIR cblas.h
        PATHS ${OPENBLAS_ROOT_DIR}/include/openblas)

if(MSVC)
    find_library(OPENBLAS_LIBRARY_RELEASE
        NAMES openblas
        PATHS ${OPENBLAS_ROOT_DIR}/lib ${OPENBLAS_ROOT_DIR}
        PATH_SUFFIXES Release)

    find_library(OPENBLAS_LIBRARY_DEBUG
        NAMES openblas
        PATHS ${OPENBLAS_ROOT_DIR}/debug/lib ${OPENBLAS_ROOT_DIR}
        PATH_SUFFIXES Debug)

    set(OpenBLAS_LIB optimized ${OPENBLAS_LIBRARY_RELEASE} debug ${OPENBLAS_LIBRARY_DEBUG})
else()
    find_library(OpenBLAS_LIB openblas)
endif()

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

mark_as_advanced(OPENBLAS_LIBRARY_DEBUG OPENBLAS_LIBRARY_RELEASE
OpenBLAS_LIB OpenBLAS_INCLUDE_DIR OPENBLAS_ROOT_DIR)
