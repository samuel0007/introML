#ifndef LinearReg_H
#define LinearReg_H


#include <eigen3/Eigen/Dense>



class LinearRegression{

    public: 
        
        double RMSE(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat);
        void GradientDescent(const Eigen::MatrixXd& X,const Eigen::MatrixXd& y, Eigen::VectorXd& omega,const float l_rate,const unsigned iterations);
        Eigen::VectorXd LeastSquare(const Eigen::MatrixXd&, const Eigen::VectorXd& y);



    private:


};



#endif