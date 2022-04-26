#ifndef LINEAR_SVM_H_
#define LINEAR_SVM_H_

#include <Eigen/Dense>
#include <Eigen/QR>
#include <fstream>
#include <iostream>
#include <string>

class LinearSVM {
 public:
  // train contains a Matrix of collected data
  // Y contains the expected ground truth
  LinearSVM(Eigen::MatrixXd X, Eigen::VectorXd y, double lambda,
            double step_size, const std::string& label) {
    const unsigned int n = X.rows();
    const unsigned int m = X.cols();

    // We define the hinge_loss
    // auto calc_hinge_loss = [K](Eigen::VectorXd y, Eigen::VectorXd w,
    //                            Eigen::VectorXd x) -> double {
    //   double sum = 0;
    //   for (auto y_i : y) {
    //     std::max(0, 1 - y_i * w.dot(x));
    //   }
    //   return sum;
    // };

    // // Defining the soft margin loss function
    // auto calc_soft_margin_loss = [lambda](Eigen::VectorXd y, Eigen::VectorXd
    // w,
    //                                       Eigen::VectorXd x, ) -> double {
    //   return 0.5 * std::pow(w.norm(), 2) + lambda * calc_hinge_loss(y, w, x);
    // };

    // We define the logistical loss
    auto calc_grad = [](double y, Eigen::VectorXd w,
                        Eigen::VectorXd x) -> Eigen::VectorXd {
      double exp_term = std::exp(y * w.dot(x));
      return -y * x / (1 + exp_term);
    };

    // We loop over the rows of the ground truth table and store the alphas
    // in the columns of the matrix A
    Eigen::VectorXd w = Eigen::VectorXd::Constant(X.cols(), 1);
    double error = 1;
    const double rtol = 1e-4;  // relative tolerance
    const double atol = 1e-5;  // absolute tolerance
    unsigned int step = 0;
    bool optimized = true;
    // We now use gradient descent to find the minimum of the loss function
    while (error > atol && error > w.norm() * rtol) {
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(X.cols());
      Eigen::VectorXd old_w = w;
      // We build the gradient
      for (unsigned int i = 0; i < X.rows(); ++i) {
        gradient += calc_grad(y[i], w, X.row(i).transpose());
      }
      gradient /= n;
      // Update the vector
      // std::cout << "Gradient :" << gradient << std::endl;
      w = w - step_size * gradient;
      error = (w - old_w).norm();
      // If we can't find an optimzation we break out of the loop
      if (step++ >= 100) {
        std::cout << label << "Could not  be optimized using gradient descent"
                  << std::endl
                  << "Final Error : " << error << std::endl;
        optimized = false;
        break;
      }
    }
    if (optimized) {
      std::cout << label << " optimized using gradient descent" << std::endl
                << "atol :" << atol << std::endl
                << "rtol :" << rtol << std::endl;
    }
    this->w_ = w;
  };

  Eigen::VectorXd predict(Eigen::MatrixXd M) {
    auto sigmoid = [](double x) { return 1. / (1 + std::exp(-x)); };
    Eigen::VectorXd p(M.rows());

    for (unsigned int i = 0; i < M.rows(); ++i) {
      p[i] = sigmoid(M.row(i).dot(w_));
    }
    this->pred_ = p;
    return p;
  }

  void print_prediction() {
    std::cout << "Prediction : " << std::endl << pred_ << std::endl;
  }

 private:
  // Stores the kernel function
  Eigen::VectorXd w_;  // Stores the function that predicts the data
  Eigen::VectorXd pred_;
  std::string label_;
};  // end SVM

#endif  // define SVM_H_
