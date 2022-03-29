#include "task1a.h"

#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/QR>
#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace std;

std::vector<int> get_folds(size_t nfolds) {
  std::vector<int> in;
  std::vector<int> out;
  static std::random_device rd;
  static std::mt19937 gen(rd());
  for (int i = 0; i < 13; ++i) in.push_back(i);
  std::shuffle(in.begin(), in.end(), gen);
  for (int i = 0; i < nfolds; ++i) out.push_back(in[i]);
  return out;
}

Eigen::MatrixXd openData(string fileToOpen) {
  // the inspiration for creating this function was drawn from here (I did NOT
  // copy and paste the code)
  // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

  // the input is the file: "fileToOpen.csv":
  // a,b,c
  // d,e,f
  // This function converts input file data into the Eigen matrix format

  // the matrix entries are stored in this variable row-wise. For example if we
  // have the matrix: M=[a b c
  //    d e f]
  // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable
  // "matrixEntries" is a row vector later on, this vector is mapped into the
  // Eigen matrix format
  vector<double> matrixEntries;

  // in this object we store the data from the matrix
  ifstream matrixDataFile(fileToOpen);

  // this variable is used to store the row of the matrix that contains commas
  string matrixRowString;

  // this variable is used to store the matrix entry;
  string matrixEntry;

  // this variable is used to track the number of rows
  int matrixRowNumber = 0;

  while (getline(matrixDataFile,
                 matrixRowString))  // here we read a row by row of
                                    // matrixDataFile and store every line into
                                    // the string variable matrixRowString
  {
    stringstream matrixRowStringStream(
        matrixRowString);  // convert matrixRowString that is a string to a
                           // stream variable.

    while (getline(matrixRowStringStream, matrixEntry,
                   ','))  // here we read pieces of the stream
                          // matrixRowStringStream until every comma, and store
                          // the resulting character into the matrixEntry
    {
      matrixEntries.push_back(stod(
          matrixEntry));  // here we convert the string to double and fill in
                          // the row vector storing all the matrix entries
    }
    matrixRowNumber++;  // update the column numbers
  }

  // here we convet the vector variable into the matrix and return the resulting
  // object, note that matrixEntries.data() is the pointer to the first memory
  // location at which the entries of the vector matrixEntries are stored;
  return Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      matrixEntries.data(), matrixRowNumber,
      matrixEntries.size() / matrixRowNumber);
}

Eigen::VectorXd RidgeRegressionLU(const Eigen::MatrixXd& X,
                                  const Eigen::VectorXd& y,
                                  const double lambda) {
  assert(X.rows() == y.rows());
  Eigen::MatrixXd A = X.transpose() * X +
                      lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols());
  Eigen::VectorXd rhs = X.transpose() * y;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
  return lu.solve(rhs);
}

Eigen::VectorXd RidgeRegressionQR(const Eigen::MatrixXd& X,
                                  const Eigen::VectorXd& y,
                                  const double lambda) {
  assert(X.rows() == y.rows());
  Eigen::MatrixXd A = X.transpose() * X +
                      lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols());
  Eigen::VectorXd rhs = X.transpose() * y;
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
  return qr.solve(rhs);
}

double RMSE(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat) {
  assert(y.size() == y_hat.size());
  return std::pow(std::pow((y - y_hat).sum(), 2), 0.5) / y.size();
}
