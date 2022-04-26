#include <Eigen/Dense>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include "linearsvm.hpp"
#include "loadcsv.hpp"
#include "ridgeregression.hpp"
#include "svm.hpp"

int main() {
  Eigen::MatrixXd X_train = openData(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task2/csv/train_zero.csv");
  Eigen::MatrixXd Y = openData(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task2/csv/train_labels_noHeader.csv");

  Eigen::MatrixXd X_test = openData(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task2/csv/test_zero.csv");

  std::vector<std::string> labels = {"pid",
                                     "LABEL_BaseExcess",
                                     "LABEL_Fibrinogen",
                                     "LABEL_AST",
                                     "LABEL_Alkalinephos",
                                     "LABEL_Bilirubin_total",
                                     "LABEL_Lactate",
                                     "LABEL_TroponinI",
                                     "LABEL_SaO2",
                                     "LABEL_Bilirubin_direct",
                                     "LABEL_EtCO2",
                                     "LABEL_Sepsis",
                                     "LABEL_RRate",
                                     "LABEL_ABPm",
                                     "LABEL_SpO2",
                                     "LABEL_Heartrate"};
  Eigen::VectorXd pids = X_test.col(0);
  X_test = X_test.block(0, 1, X_test.rows(), X_test.cols() - 1);
  X_train = X_train.block(0, 1, X_train.rows(), X_train.cols() - 1);
  Eigen::MatrixXd Y_classification = Y.block(0, 1, Y.rows(), 11);
  Eigen::MatrixXd Y_regression = Y.block(0, 12, Y.rows(), 4);

  const double scaling_factor = 1e-2;
  X_test *= scaling_factor;
  X_train *= scaling_factor;

  const double tau = 0.1;
  const double step_size = 0.75;
  const double lambda_class = 0.25;
  const double lambda_reg = 0.01;

  // kernel functions for non-linear SVM
  auto poly_2d = [](const Eigen::VectorXd vc1,
                    const Eigen::VectorXd vc2) -> double {
    return std::pow(vc1.dot(vc2) + 1, 2);
  };

  auto gaussian = [tau](const Eigen::VectorXd vc1,
                        const Eigen::VectorXd vc2) -> double {
    return std::exp(-std::pow((vc1 - vc2).norm(), 2) / (2 * std::pow(tau, 2)));
  };
  auto k = gaussian;

  // NONLinear SVM is not fully implemented
  // SVM<decltype(k)> svm(train, label, k, step_size);

  Eigen::MatrixXd P(X_test.rows(), Y.cols());
  P.col(0) = pids;
  std::mutex P_mutex;

  Eigen::MatrixXd A(X_train.rows(), Y_classification.cols());
  std::mutex A_mutex;

  Eigen::MatrixXd K(X_train.rows(), X_test.rows());
  std::mutex K_mutex;

  auto init_svm = [&k, &lambda_class, &step_size, &A, &A_mutex, &X_train,
                   &X_test, &labels](Eigen::VectorXd y, int ind) {
    std::cout << "Start training of " << labels[ind] << std::endl;
    SVM_SGD<decltype(k)> svm(X_train, y, k, lambda_class, step_size,
                             labels[ind]);
    const std::lock_guard<std::mutex> lock(A_mutex);
    std::cout << "Adding alpha for " << labels[ind] << std::endl;
    A.col(ind) = svm.get_alpha();
  };

  auto build_K_column = [&X_train, &k, &K, &K_mutex, &X_test](int ind) {
    Eigen::VectorXd k_term(X_train.rows());
    for (unsigned int i = 0; i < k_term.size(); ++i)
      k_term[i] = k(X_train.row(i), X_test.row(ind));
    // std::cout << "Writing to K column  " << ind << std::endl;
    const std::lock_guard<std::mutex> lock(K_mutex);
    K.col(ind) = k_term;
    //   std::cout << "K column " << ind << " done" << std::endl;
  };

  auto init_regression = [&lambda_reg, &P, &P_mutex, &X_train, &X_test,
                          &labels](Eigen::VectorXd y, int ind) {
    std::cout << "Start training of " << labels[ind] << std::endl;
    Regression_Model reg(X_train, y, lambda_reg);
    const Eigen::VectorXd p = reg.predict(X_test);
    const std::lock_guard<std::mutex> lock(P_mutex);
    std::cout << "Adding prediction for " << labels[ind] << std::endl;
    P.col(ind) = p;
    std::cout << "Added prediction for " << labels[ind] << std::endl;
  };

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < Y_classification.cols(); ++i) {
    threads.push_back(std::thread(init_svm, Y_classification.col(i), i));
  }
  for (unsigned int i = 0; i < Y_regression.cols(); ++i) {
    threads.push_back(std::thread(init_regression, Y_regression.col(i),
                                  i + Y_classification.cols() + 1));
  }
  for (auto& thread : threads) {
    thread.join();
  }

  int nthreads = 0;
  int max_threads = 50;
  std::vector<std::thread> K_threads;
  std::cout << "Started building K" << std::endl;
  while (nthreads < X_test.rows()) {
    for (int i = 0; i < max_threads; ++i) {
      if (nthreads < X_test.rows())
        K_threads.push_back(std::thread(build_K_column, nthreads));
      ++nthreads;
    }

    for (auto& thread : K_threads) {
      thread.join();
    }
    K_threads.clear();

    if (nthreads % (X_test.rows() / 100) == 0) {
      std::cout << nthreads / (X_test.rows() / 100) << "% of K done"
                << std::endl;
    }
  }

  auto sigmoid = [&k](double x) { return 1. / (1 + std::exp(-x)); };

  for (unsigned int i = 0; i < P.rows(); ++i) {
    for (unsigned int j = 0; j < Y_classification.cols(); ++j) {
      P(i, j + 1) = sigmoid(K.col(i).dot(A.col(j)));
    }
  }

  std::cout << "Writing prediction to csv" << std::endl;
  save_pred(labels, P);

  return 0;
}
