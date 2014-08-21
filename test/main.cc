/* some simple test cases for the NNLS code. */

#include <iostream>

//#define EIGEN3_NNLS_DEBUG 1
#include <Eigen/Eigen>
#include "nnls.h"

using namespace Eigen;

// Trivial unit tests with cases taken from
// http://www.turkupetcentre.net/reports/tpcmod0020_app_a.pdf


template <typename MatrixType>
bool testNNLS(const MatrixType &A,
              const Matrix<typename MatrixType::Scalar, MatrixType::RowsAtCompileTime, 1> &b,
              const Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, 1> &x,
              typename MatrixType::Scalar eps=1e-10)
{
  NNLS<MatrixType> nnls(A, 30, eps);
  if (! nnls.solve(b)) {
    std::cerr << __FILE__ << ": Convergence failed!" << std::endl; return false;
  }
  Array<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, 1> err
      = (x-nnls.x()).array().abs();
  if (err.maxCoeff() > 1e-6){
    std::cerr << __FILE__ << ": Precision error: Expected (" << x.transpose()
              << ") got: (" << nnls.x().transpose()
              << "), err: " << err.maxCoeff() << std::endl;
    return false;
  }

  if (! nnls.check(b)) {
    std::cerr << __FILE__ << ": check() KKT returned false!" << std::endl; return false;
  }

  return true;
}

// 4x2 problem, unconstrained solution positive
bool case_1 () {
  Matrix<double, 4, 2> A(4,2);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 2, 1> x(2);
  A << 1, 1,  2, 4,  3, 9,  4, 16;
  b << 0.6, 2.2, 4.8, 8.4;
  x << 0.1, 0.5;

  return testNNLS(A, b, x);
}

// 4x3 problem, unconstrained solution positive
bool case_2 () {
  Matrix<double, 4, 3> A(4,3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);

  A << 1,  1,  1,
       2,  4,  8,
       3,  9, 27,
       4, 16, 64;
  b << 0.73, 3.24, 8.31, 16.72;
  x << 0.1, 0.5, 0.13;

  return testNNLS(A, b, x);
}

// Simple 4x4 problem, unconstrained solution non-negative
bool case_3 () {
  Matrix<double, 4, 4> A(4, 4);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 4, 1> x(4);

  A <<1,  1,  1,   1,
      2,  4,  8,  16,
      3,  9, 27,  81,
      4, 16, 64, 256;
  b << 0.73, 3.24, 8.31, 16.72;
  x << 0.1, 0.5, 0.13, 0;

  return testNNLS(A, b, x);
}

// Simple 4x3 problem, unconstrained solution non-negative
bool case_4 () {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);

  A <<1,  1,  1,
      2,  4,  8,
      3,  9, 27,
      4, 16, 64;
  b << 0.23, 1.24, 3.81, 8.72;
  x << 0.1, 0, 0.13;

  return testNNLS(A, b, x);
}

// Simple 4x3 problem, unconstrained solution indefinite
bool case_5 () {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);

  A <<1,  1,  1,
      2,  4,  8,
      3,  9, 27,
      4, 16, 64;
  b << 0.13, 0.84, 2.91, 7.12;
  // Solution obtained by original nnls() implementation
  // in Fortran
  x << 0.0, 0.0, 0.1106544;

  return testNNLS(A, b, x);
}

// 200x100 random problem
bool case_6 () {
  MatrixXd A(200,100); A.setRandom();
  VectorXd b(200); b.setRandom();
  VectorXd x(100);

  return NNLS<MatrixXd>::solve(A, b, x);
}


int main(int argc, char *argv[])
{
  // Run test cases...
  bool ok = true;
  ok &= case_1();
  ok &= case_2();
  ok &= case_3();
  ok &= case_4();
  ok &= case_5();
  ok &= case_6();

  if (ok) return 0;
  else return -1;

}
