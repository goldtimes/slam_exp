#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

int main(int argc, char** argv) {
    auto qua = Eigen::AngleAxisd((3.0 / M_PI), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd((3.0 / M_PI), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd((3.0 / M_PI), Eigen::Vector3d::UnitX());
    auto R = qua.toRotationMatrix();

    std::cout << "qua: " << qua.x() << " " << qua.y() << " " << qua.z() << " " << qua.w() << std::endl;

    auto qua_coj = qua.conjugate();

    std::cout << "qua coj: " << qua_coj.x() << " " << qua_coj.y() << " " << qua_coj.z() << " " << qua_coj.w()
              << std::endl;

    std::cout << "qua coj to rotation: " << qua_coj.toRotationMatrix() << std::endl;
    std::cout << "R inv" << R.inverse() << std::endl;
    auto R_tran = R.transpose();
    std::cout << "R_tran: \n" << R_tran << std::endl;

    // std::cout << "R: \n" << R << std::endl;

    // auto R_coj = R.conjugate();
    // std::cout << "R_conj: \n" << R_coj << std::endl;

    // auto R_tran = R.transpose();

    // std::cout << "R_tran: \n" << R_coj << std::endl;

    // std::cout << "R * R_tran: \n" << R * R_tran << std::endl;
    Eigen::Vector3d p(1, 2, 3);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(p);
    Eigen::MatrixXd Q = qr.householderQ();
    std::cout << "Q:" << Q << std::endl;
    std::cout << "R:" << qr.hCoeffs() << std::endl;

    return 0;
}