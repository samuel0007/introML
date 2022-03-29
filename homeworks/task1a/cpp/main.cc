// https:aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <vector>

#include "task1a.h"

int main(int argc, char **argv) {
  Eigen::MatrixXd dataMat = openData(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task1a/csv/train.csv");

  // Preprocessing the Data
  // Taking the last 10 rows as test
  int rows = dataMat.rows() - 10;
  int cols = dataMat.cols() - 1;
  Eigen::VectorXd y_train = dataMat.col(0).head(rows);
  Eigen::VectorXd y_test = dataMat.col(0).tail(10);

  // Defining the number of folds
  unsigned int nfolds = 10;
  int D_rows = rows / nfolds;

  std::vector<double> results(5);

  Eigen::MatrixXd train = dataMat.block(0, 1, rows, cols);
  Eigen::MatrixXd test = dataMat.block(nfolds * D_rows, 1, 10, cols);
  std::vector<Eigen::MatrixXd> partition_D;
  std::vector<Eigen::VectorXd> partition_y;
  for (unsigned int i = 0; i < nfolds; ++i) {
    partition_y.push_back(y_train.segment(i * D_rows, D_rows));
    partition_D.push_back(train.block(i * D_rows, 0, D_rows, cols));
  }

  // Defining lamdas as alphas
  Eigen::VectorXd alphas(5);
  alphas << 0.1, 1, 10, 100, 200;

  // Making aking a vector of functions
  std::vector<
      std::function<Eigen::VectorXd(Eigen::MatrixXd, Eigen::VectorXd, double)>>
      functions;
  functions.push_back(RidgeRegressionLU);
  functions.push_back(RidgeRegressionQR);

  // Looping over the alphas
  for (unsigned int i = 0; i < alphas.size(); ++i) {
    double alpha = alphas[i];
    std::vector<double> cross_validation_error(2);

    // Looping over the diffrent functions
    for (unsigned int n = 0; n < functions.size(); ++n) {
      auto phi = functions[n];
      // Looping over the folds
      for (unsigned int k = 0; k < nfolds; ++k) {
        Eigen::MatrixXd D_prime = partition_D[k];
        Eigen::VectorXd y_prime = partition_y[k];
        Eigen::MatrixXd D(rows - D_rows, cols);
        Eigen::VectorXd y_D(rows - D_rows);

        for (unsigned int j = 0; j < nfolds - 1; ++j) {
          y_D.segment(j * D_rows, D_rows) = partition_y[(j + k + 1) % nfolds];
          D.block(j * D_rows, 0, D_rows, cols) =
              partition_D[(j + k + 1) % nfolds];
        }
        // Computing the diffrent cross_validations and adding them to the cross
        // validation
        Eigen::VectorXd w;
        w = phi(D, y_D, alpha);
        cross_validation_error[n] += RMSE(D_prime * w, y_prime);
        // std::cout << cross_validation_error[n] << std::endl;
      }
      std ::cout << cross_validation_error[n] << std::endl;
    }
    // Choosing the best function

    // Making an estimate with the best function and writing into result
    int min_index = std::min_element(cross_validation_error.begin(),
                                     cross_validation_error.end()) -
                    cross_validation_error.begin();
    auto f_best = functions[min_index];
    Eigen::VectorXd w = f_best(train, y_train, alpha);
    results[i] = RMSE(y_test, test * w);
  }
  std::ofstream out(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task1a/csv/sol.csv");

  for (unsigned int i = 0; i < alphas.size(); ++i) {
    out << results[i] << std::endl;
  }
  out.close();
  return 0;
}