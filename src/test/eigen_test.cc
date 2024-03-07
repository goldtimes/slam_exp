#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

int main(int argc, char** argv) {
    auto R = Eigen::AngleAxisd((3.0 / M_PI), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd((3.0 / M_PI), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd((3.0 / M_PI), Eigen::Vector3d::UnitX()).toRotationMatrix();
    std::cout << "R: \n" << R << std::endl;

    auto R_coj = R.conjugate();
    std::cout << "R_conj: \n" << R_coj << std::endl;

    auto R_tran = R.transpose();

    std::cout << "R_tran: \n" << R_coj << std::endl;

    std::cout << "R * R_tran: \n" << R * R_tran << std::endl;

    return 0;
}