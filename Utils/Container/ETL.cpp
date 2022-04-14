#include "ETL.hpp"

#include <stdlib.h>

#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

ETL(const std::string &data, const std::string &seperator, bool head)
    : dataset(data), delimiter(seperator), header(head) {}

std::vector<std::vector<std::string>> ETL::readCSV() {
  std::ifstream file(self->dataset);
  std::vector<std::vector<std::string>> dataString;
  std::string line = "";

  while (getline(file, line)) {
    std::vector<std::string> vec;
    boost::algorithm::split(vec, line, boost::is_any_of(self->delimiter));
    dataString.push_back(vec);
  }

  file.close();
  return dataString;
}

Eigen::MatrixXd ETL::CSVtoEigen(
    const std::vector<std::vector<std::string>> &dataset, int rows, int cols) {
  if (self->header) rows = rows - 1;
  Eigen::MatrixXd mat(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat(i, j) = atof(dataset[i][j].c_str());
    }
  }
  return mat;
}

auto ETL::Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean()) {
  return data.colwise().mean();
}

auto ETL::Std(Eigen::MatrixXd data)
    -> decltype((data.array().square().colwise().sum() / (data.rows() - 1))
                    .sqrt()) {
  return ((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt();
}

Eigen::MatrixXd ETL::Normalize(Eigen::MatrixXd data, bool normalizeTarget) {
  auto mean = Mean(data);
  Eigen::MatrixXd scaled_data = data.rowwise() - mean;
  auto std = Std(scaled_data);

  Eigen::MatrixXd norm = scaled_data.array().rowwise() / std;

  return norm;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
ETL::TrainTestSplit(Eigen::MatrixXd data, float train_size) {
  int rows = data.rows();
  int train_rows = round(train_size * rows);
  int test_rows = rows - train_rows;

  Eigen::MatrixXd train = data.topRows(train_rows);

  Eigen::MatrixXd X_train = train.leftCols(data.cols() - 1);
  Eigen::MatrixXd y_train = train.rightCols(1);

  Eigen::MatrixXd test = data.bottomRows(test_rows);

  Eigen::MatrixXd X_test = test.leftCols(data.cols() - 1);
  Eigen::MatrixXd y_test = test.rightCols(1);

  return std::make_tuple(X_train, y_train, X_test, y_test);
}

void ETL::VectorToFile(std::vector<float> vector, std::string filename) {
  std::ofstream output_file(filename);
  std::ostream_iterator<float> output_iterator(output_file, "\n");
  std::copy(vector.begin(), vector.end(), output_iterator);
}

void ETL::EigenToFile(Eigen::MatrixXd data, std::string filename) {
  std::ofstream output_file(filename);
  if (output_file.is_open()) output_file << data << "\n";
}
