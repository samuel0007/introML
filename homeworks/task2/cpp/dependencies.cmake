# Add your custom dependencies here:

# DIR will be provided by the calling file.

set(SOURCES
  ${DIR}/main.cc
  ${DIR}/svm.hpp
  ${DIR}/linearsvm.hpp
  ${DIR}/ridgeregression.hpp
  ${DIR}/loadcsv.hpp
  ${DIR}/loadcsv.cc

)

set(LIBRARIES
  Eigen3::Eigen
)
