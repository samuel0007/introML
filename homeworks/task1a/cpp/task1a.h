#ifndef TASK1A_H
#define TASK1A_H

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

std::vector<int> get_folds(size_t nfolds);

Eigen::MatrixXd openData(string);

Eigen::VectorXd RidgeRegressionLU(const Eigen::MatrixXd&,
                                  const Eigen::VectorXd& y, const double);

Eigen::VectorXd RidgeRegressionQR(const Eigen::MatrixXd&,
                                  const Eigen::VectorXd& y, const double);

double RMSE(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat);

#endif