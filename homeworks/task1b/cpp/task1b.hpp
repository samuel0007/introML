#ifndef TASK1B_H
#define TASK1B_H

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

std::vector<int> get_folds(std::size_t nfolds);

Eigen::MatrixXd openData(std::string);

Eigen::VectorXd RidgeRegressionLU(const Eigen::MatrixXd&,
                                  const Eigen::VectorXd& y, const double);

Eigen::VectorXd RidgeRegressionQR(const Eigen::MatrixXd&,
                                  const Eigen::VectorXd& y, const double);

double RMSE(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat);

#endif
