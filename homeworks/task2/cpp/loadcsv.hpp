#ifndef LOAD_CSV_H_
#define LOAD_CSV_H_

#include <Eigen/Dense>
#include <string>
#include <vector>
Eigen::MatrixXd openData(std::string fileToOpen);

void save_pred(std::vector<std::string> labels, Eigen::MatrixXd P);
#endif  // LOAD_CSV_H
