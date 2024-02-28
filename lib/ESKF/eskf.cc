#include "eskf.hh"
#include "SO3.hh"

namespace eskf {
void ESKF::Predict(const IMU& imu_data, double dt) {
    auto imu = imu_data;
    // 加速度和角速度减去偏移
    imu.acc -= state.ba;
    imu.gro -= state.bg;
    // 记录上一个状态的角度
    Eigen::Matrix3d r = state.q.toRotationMatrix();
    // 更新状态量的旋转部分
    state.q = Eigen::Quaterniond(r * so3Exp(imu.gro * dt));
    state.q.normalize();
    // 更新p
    state.p += state.v * dt;
    // 更新v
    state.v += (r * imu.acc + state.g) * dt;

    // 协方差的传播
    Eigen::Matrix<double, 18, 18> Fx;
    Fx.setIdentity();
    Eigen::Matrix<double, 18, 18> Fw;
    Fw.setZero();

    Fx.block<3, 3>(0, 0) = so3Exp(-1 * imu.gro * dt);
    ///.paper
    // Fx.block<3,3>(0,9) = -1*A_T(imu.gro*dt)*dt;
    ///.fastlio
    Fx.block<3, 3>(0, 9) = -1 * A_T(-imu.gro * dt) * dt;

    Fx.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
    Fx.block<3, 3>(6, 0) = r * skew(imu.acc) * dt * (-1);
    // Fx.block<3,3>(6,0) =  skew_symmetric(rotation*imu.acc)*dt*(-1);
    Fx.block<3, 3>(6, 12) = r * dt * (-1);
    Fx.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity() * dt;

    ///. 近似做法
    // Fw.block<3,3>(0,0) = -Eigen::Matrix3d::Identity()*dt;
    ///. fast_lio做法
    Fw.block<3, 3>(0, 0) = -1 * A_T(-imu.gro * dt) * dt;
    ///. paper
    // Fw.block<3,3>(0,0) = -1*A_T(imu.gro*dt)*dt;
    Fw.block<3, 3>(6, 3) = -1 * rotation * dt;
    Fw.block<3, 3>(9, 6) = Fw.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity() * dt;

    P = Fx * P * Fx.transpose() + Fw * Q * Fw.transpose();
}
void ESKF::Update() {
    auto x_k_k = state;
    auto x_k_last = state;

    // 利用观测来更新 计算k
    Eigen::MatrixXd k;
    Eigen::MatrixXd H_k;

    Eigen::Matrix<double, 18, 18> P_in_update = P;
    Eigen::MatrixXd z_k;
    Eigen::MatrixXd R_inv;  // R 直接写死0.001;
    zhr_model(x_k_k, z_k, H_k, R_inv);

    ///.  计算K
    Eigen::MatrixXd H_kt = H_k.transpose();
    // K = (H_kt*R_inv*H_k+P_in_update.inverse()).inverse()*H_kt*R_inv;
    K = (H_kt * H_k + (P_in_update / 0.001).inverse()).inverse() * H_kt;
    Eigen::MatrixXd update_x = K * z_k;

    x_k_k.rotation = x_k_k.rotation.toRotationMatrix() * so3Exp(update_x.block<3, 1>(0, 0));
    x_k_k.rotation.normalize();
    x_k_k.position = x_k_k.position + update_x.block<3, 1>(3, 0);
    x_k_k.velocity = x_k_k.velocity + update_x.block<3, 1>(6, 0);
    x_k_k.bg = x_k_k.bg + update_x.block<3, 1>(9, 0);
    x_k_k.ba = x_k_k.ba + update_x.block<3, 1>(12, 0);
    x_k_k.gravity = x_k_k.gravity + update_x.block<3, 1>(15, 0);
    state = x_k_k;

    P = (Eigen::Matrix<double, 18, 18>::Identity() - K * H_k) * P_in_update;
}

Eigen::Matrix3d ESKF::A_T(const Eigen::Vector3d& v) {
    Eigen::Matrix3d res;
    double squaredNorm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    double norm = std::sqrt(squaredNorm);
    if (norm < 1e-11) {
        res = Eigen::Matrix3d::Identity();
    } else {
        res = Eigen::Matrix3d::Identity() + (1 - std::cos(norm)) / squaredNorm * skew(v) +
              (1 - std::sin(norm) / norm) / squaredNorm * skew(v) * skew(v);
    }
    return res;
}

Eigen::Matrix<double, 18, 1> ESKF::getErrorState18(const ESKF::State& s1, const ESKF::State& s2) {
    Eigen::Matrix<double, 18, 1> es;
    es.setZero();
    es.block<3, 1>(0, 0) = SO3Log(s2.q.toRotationMatrix().transpose() * s1.q.toRotationMatrix());
    es.block<3, 1>(3, 0) = s1.p - s2.p;
    es.block<3, 1>(6, 0) = s1.v - s2.v;
    es.block<3, 1>(9, 0) = s1.bg - s2.bg;
    es.block<3, 1>(12, 0) = s1.ba - s2.ba;
    es.block<3, 1>(15, 0) = s1.g - s2.g;
    return es;
}

}  // namespace eskf