#ifndef ETL_h
#define ETL_h

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class ETL {
  std::string dataset;
  std::string delimiter;
  bool header;

 public:
  ETL(const std::string &data, const std::string &seperator, bool head)
      : dataset(data), delimiter(seperator), header(head) {}
  std::vector<std::vector<std::string> > readCSV();
  Eigen::MatrixXd CSVtoEigen(const std::vector<std::vector<std::string> > &,
                             int, int);
  Eigen::MatrixXd Normalize(Eigen::MatrixXd, bool);
  auto Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean());
  auto Std(Eigen::MatrixXd data)
      -> decltype((data.array().square().colwise().sum() / (data.rows() - 1))
                      .sqrt());
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
  TrainTestSplit(Eigen::MatrixXd, float);
  void VectorToFile(std::vector<float>, std::string);
  void EigenToFile(Eigen::MatrixXd, std::string);
};

#endif
