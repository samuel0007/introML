#include "LinearRegression.h"


void LinearRegression::GradientDescent(const Eigen::MatrixXd& X,const Eigen::MatrixXd& y, Eigen::VectorXd& omega,const float l_rate,const unsigned iterations){

    Eigen::VectorXd temp_omega = omega;
    std::size_t n = X.rows();
    std::size_t m = X.cols();

    for(unsigned iteration = 0; iteration < iteration; ++iteration){
        temp_omega = (Eigen::MatrixXd::Identity(n,m) - l_rate*X.transpose()*X)*temp_omega+l_rate*X.transpose()*y;
    }
    omega = temp_omega;
}
Eigen::VectorXd LinearRegression::LeastSquare(const Eigen::MatrixXd& X, const Eigen::VectorXd& y){
    assert(X.rows() == y.rows());
    return X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
}
double LinearRegression::RMSE(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat){
    assert(y.size() == y_hat.size());
    return std::pow(std::pow((y-y_hat).sum(),2),0.5)/y.size();
}