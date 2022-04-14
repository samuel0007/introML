
// https:aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
#include <Eigen/Dense>
#include <Eigen/QR>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "task1b.hpp"

int main() {
  Eigen::MatrixXd dataMat = openData(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task1b/csv/train.csv");

  // Preprocessing the Data
  // Taking the last 10 rows as test
  int rows = dataMat.rows();
  int cols = dataMat.cols() - 2;
  Eigen::VectorXd y = dataMat.col(1);
  Eigen::VectorXd ID = dataMat.col(0);

  Eigen::MatrixXd train = dataMat.block(0, 2, rows, cols);

  std::cout << "Building the functions" << std::endl;
  auto quadr = [&](Eigen::VectorXd x) { return x.array().square(); };
  auto exponent = [&](Eigen::VectorXd x) { return x.array().exp(); };
  auto cosine = [&](Eigen::VectorXd x) { return x.array().cos(); };
  Eigen::MatrixXd X(rows, 21);
  X.block(0, 0, rows, 5) = train;
  std::cout << "Building X" << std::endl;
  for (unsigned int i = 0; i < 5; ++i) {
    X.col(i + 5) = quadr(train.col(i));
    X.col(i + 10) = exponent(train.col(i));
    X.col(i + 15) = cosine(train.col(i));
  }
  X.col(20) = Eigen::VectorXd::Constant(rows, 1);

  // std::cout << "Building QR" << std::endl;

  // Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
  // std::cout << "Solving X" << std::endl;
  // Eigen::VectorXd w = qr.solve(y);
  // std::cout << "Writing to csv" << std::endl;

  std::cout << "Solving X" << std::endl;

  Eigen::VectorXd w = RidgeRegressionQR(X, y, 0.05);

  std::cout << "Writing to csv" << std::endl;
  std::ofstream out(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task1b/csv/sol.csv");

  for (unsigned int i = 0; i < 21; ++i) {
    out << w[i] << std::endl;
  }
  out.close();
}
