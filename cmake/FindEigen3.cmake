#
# Try to find Eigen3 include dirs.
#
# Eigen is a C++ template library, therefore there are no libraries to find.
#

find_path(EIGEN3_INCLUDE_DIR Eigen/Eigen 
          PATH_SUFFIXES eigen3)

set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3  DEFAULT_MSG
                                  EIGEN3_INCLUDE_DIR)

mark_as_advanced(EIGEN3_INCLUDE_DIR EIGEN3_LIBRARY)
