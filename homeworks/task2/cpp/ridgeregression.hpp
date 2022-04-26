#ifndef RIDGE_REGRESSION_H
#define RIDGE_REGRESSION_H

#include <Eigen/Dense>
#include <Eigen/QR>
#include <cassert>

class Regression_Model {
 public:
  Regression_Model(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                   const double lambda) {
    assert(X.rows() == y.rows());
    Eigen::MatrixXd A = X.transpose() * X +
                        lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols());
    Eigen::VectorXd rhs = X.transpose() * y;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    w_ = qr.solve(rhs);
  }

  Eigen::VectorXd predict(Eigen::MatrixXd X_test) {
    p_ = X_test * w_;
    return p_;
  }

 private:
  Eigen::VectorXd w_;
  Eigen::VectorXd p_;
};
#endif
