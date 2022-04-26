#include "loadcsv.hpp"

#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/QR>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

Eigen::MatrixXd openData(std::string fileToOpen) {
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
  std::vector<double> matrixEntries;

  // in this object we store the data from the matrix
  std::ifstream matrixDataFile(fileToOpen);

  // this variable is used to store the row of the matrix that contains commas
  std::string matrixRowString;

  // this variable is used to store the matrix entry;
  std::string matrixEntry;

  // this variable is used to track the number of rows
  int matrixRowNumber = 0;

  while (std::getline(
      matrixDataFile,
      matrixRowString))  // here we read a row by row of
                         // matrixDataFile and store every line into
                         // the string variable matrixRowString
  {
    std::stringstream matrixRowStringStream(
        matrixRowString);  // convert matrixRowString that is a string to a
                           // stream variable.

    while (std::getline(
        matrixRowStringStream, matrixEntry,
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

void save_pred(std::vector<std::string> labels, Eigen::MatrixXd P) {
  std::ofstream out(
      "/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/"
      "homeworks/task2/csv/solution.csv");
  for (unsigned int i = 0; i < labels.size(); ++i) {
    if (i != 0) out << ',';
    out << labels[i];
  }
  out << std::endl;
  for (unsigned int i = 0; i < P.rows(); ++i) {
    for (unsigned int j = 0; j < P.cols(); ++j) {
      if (j == 0)
        out << P(i, j);
      else
        out << ',' << P(i, j);
    }
    out << std::endl;
  }
  out.close();
};
