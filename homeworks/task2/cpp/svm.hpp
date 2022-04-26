#ifndef SVM_H_
#define SVM_H_

#include <Eigen/Dense>
#include <Eigen/QR>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

template <typename F>
class SVM {
 public:
  // train contains a Matrix of collected data
  // Y contains the expected ground truth
  SVM(Eigen::MatrixXd X, Eigen::VectorXd y, F k, double lambda,
      double step_size, const std::string &label) {
    const unsigned int n = X.rows();
    const unsigned int m = X.cols();
    std::cout << "Building K for " << label << std::endl;
    Eigen::MatrixXd K(n, n);
    for (unsigned int i = 0; i < n; ++i) {
      for (unsigned int j = 0; j < i + 1; ++j) {
        double tmp = k(X.row(i), X.row(j));
        K(i, j) = tmp;
        K(j, i) = tmp;
      }
      if (i % (n / 100) == 0)
        std::cout << i / (n / 100) << "% building K_train for " << label
                  << " done" << std::endl;
    }

    auto calc_grad = [&K](double y, Eigen::VectorXd w,
                          int ind) -> Eigen::VectorXd {
      double exp_term = std::exp(y * w.dot(K.col(ind)));
      return -y * K.col(ind) / (1 + exp_term);
    };

    // We loop over the rows of the ground truth table and store the alphas
    // in the columns of the matrix A
    Eigen::VectorXd w = Eigen::VectorXd::Constant(X.rows(), 1);
    double error = 1;
    const double rtol = 1e-3;  // relative tolerance
    const double atol = 1e-3;  // absolute tolerance
    unsigned int step = 0;
    bool optimized = true;
    // We now use gradient descent to find the minimum of the loss function
    std::cout << "Starting Gradient descent for " << label << std::endl;
    while (error > atol && error > w.norm() * rtol) {
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(X.rows());
      Eigen::VectorXd old_w = w;
      // We build the gradient
      for (unsigned int i = 0; i < X.rows(); ++i) {
        gradient += calc_grad(y[i], w, i);
      }
      gradient /= n;
      // Update the vector
      // std::cout << "Gradient :" << gradient << std::endl;
      w = w - step_size * gradient;
      error = (w - old_w).norm();
      // If we can't find an optimzation we break out of the loop
      if (step++ >= 50) {
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
    this->X_ = X;
  };

  Eigen::VectorXd predict(Eigen::MatrixXd M, F k) {
    auto sigmoid = [&k](double x) { return 1. / (1 + std::exp(-x)); };
    Eigen::VectorXd p(M.rows());
    Eigen::MatrixXd K(X_.rows(), M.rows());
    int n = K.rows();
    for (unsigned int i = 0; i < K.rows(); ++i) {
      for (unsigned int j = 0; j < K.cols(); ++j) {
        K(i, j) = k(X_.row(i), M.row(j));
      }
      if (i % (n / 100) == 0)
        std::cout << i / (n / 100) << "% building K_test for " << label_
                  << " done" << std::endl;
    }
    for (unsigned int i = 0; i < M.rows(); ++i) {
      p[i] = sigmoid(K.col(i).dot(w_));
    }
    this->pred_ = p;
    return p;
  }

  void print_prediction() {
    std::cout << "Prediction : " << std::endl << pred_ << std::endl;
  }

 private:
  // Stores the kernel function
  Eigen::VectorXd w_;     // Stores the function that predicts the data
  Eigen::VectorXd pred_;  // Stores the prediction
  Eigen::MatrixXd X_;     // Stores the training data
  std::string label_;     // Stores the label
};                        // end SVM

// Here we use Stochastic Gradient Descent insted of the normal GD
template <typename F>
class SVM_SGD {
 public:
  // train contains a Matrix of collected data
  // Y contains the expected ground truth
  SVM_SGD(Eigen::MatrixXd X, Eigen::VectorXd y, F k, double lambda,
          double step_size, const std::string &label) {
    const unsigned int n = X.rows();
    &const unsigned int m = X.cols();

    auto calc_grad = [&k, X](double y, Eigen::VectorXd alpha,
                             Eigen::VectorXd x) -> Eigen::VectorXd {
      Eigen::VectorXd k_term(alpha.size());
      for (unsigned int i = 0; i < X.rows(); ++i) k_term[i] = k(X.row(i), x);
      double exp_term = std::exp(y * alpha.dot(k_term));
      return -y * k_term / (1 + exp_term);
    };

    // We loop over the rows of the ground truth table and store the alphas
    // in the columns of the matrix A
    Eigen::VectorXd alpha = Eigen::VectorXd::Constant(X.rows(), 1);
    double error = 1;
    const double rtol = 1e-3;  // relative tolerance
    const double atol = 1e-3;  // absolute tolerance
    unsigned int step = 0;
    const unsigned int mini_batch_size = 100;
    const unsigned int max_step = 1000;
    bool optimized = true;

    std::random_device rd;   // obtain a random number from hardware
    std::mt19937 gen(rd());  // seed the generator
    std::uniform_int_distribution<> distr(0, X.rows());  // define the range
    // We now use gradient descent to find the minimum of the loss function
    std::cout << "Starting Gradient descent for " << label << std::endl;
    while (error > atol && error > alpha.norm() * rtol) {
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(X.rows());
      Eigen::VectorXd old_alpha = alpha;
      // We build the gradient

      for (unsigned int i = 0; i < mini_batch_size; ++i) {
        int random_number = distr(gen);
        gradient += calc_grad(y[random_number], alpha, X.row(random_number));
      }
      gradient /= mini_batch_size;
      // Update the vector
      // std::cout << "Gradient :" << gradient << std::endl;
      alpha = alpha - step_size * gradient;
      error = (alpha - old_alpha).norm();
      // If we can't find an optimzation we break out of the loop
      if (step++ >= max_step) {
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
    this->alpha_ = alpha;
    this->X_ = X;
    this->label_ = label;
  };

  Eigen::VectorXd predict(Eigen::MatrixXd M, F k) {
    auto sigmoid = [&k](double x) { return 1. / (1 + std::exp(-x)); };
    Eigen::VectorXd p(M.rows());
    int n = M.rows();
    for (unsigned int i = 0; i < n; ++i) {
      Eigen::VectorXd k_term(alpha_.size());
      for (unsigned int j = 0; j < X_.rows(); ++j)
        k_term[j] = k(X_.row(j), M.row(i));
      p[i] = sigmoid(k_term.dot(alpha_));
      if (i % (n / 100) == 0)
        std::cout << i / (n / 100) << "% building K_test for " << label_
                  << " done" << std::endl;
    }
    this->pred_ = p;
    return p;
  }

  Eigen::VectorXd get_alpha() { return this->alpha_; }

 private:
  // Stores the kernel function
  Eigen::VectorXd w_;      // Stores the function that predicts the data
  Eigen::VectorXd alpha_;  // THe solution of the Dual Problem
  Eigen::VectorXd pred_;   // Stores the prediction
  Eigen::MatrixXd X_;      // Stores the training data
  std::string label_;      // Stores the label
};

#endif
