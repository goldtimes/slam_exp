#pragma once
#include <Eigen/Dense>
#include <iostream>
#include "imu.hh"

namespace eskf {
class ESKF {
   public:
    struct State {
        Eigen::Quaterniond q;
        Eigen::Vector3d p;
        Eigen::Vector3d v;
        Eigen::Vector3d ba;
        Eigen::Vector3d bg;
        Eigen::Vector3d g;

        State() {
            q = Eigen::Quaterniond::Identity();
            p = Eigen::Vector3d::Zero();
            v = Eigen::Vector3d::Zero();
            ba = Eigen::Vector3d::Zero();
            bg = Eigen::Vector3d::Zero();
            g = Eigen::Vector3d::Zero();
        }
        friend std::ostream& operator<<(std::ostream& os, const State& state) {
            os << "rotation:" << state.q.coeffs().transpose() << "\nposition:" << state.p.transpose()
               << "\nvelocity:" << state.v.transpose() << "\nba:" << state.ba.transpose()
               << "\nbg:" << state.bg.transpose() << "\ng:" << state.g.transpose() << std::endl;
            return os;
        }
    };

   public:
    ESKF() {
    }
    ~ESKF() {
    }

    void Predict(const IMU& imu, double dt);
    void Update();
    Eigen::Matrix3d A_T(const Eigen::Vector3d& v);
    Eigen::Matrix<double, 18, 1> getErrorState18(const ESKF::State& s1, const ESKF::State& s2);

    State GetStat
#include "lib/ESKF/eskf.hh" e() {
        return state;
} State GetLastState() {
    return last_state;
}
void SetState(const State& in_state) {
    state = in_state;
}

Eigen::Matrix<double, 18, 18> GetP() {
    return P;
}
Eigen::Matrix<double, 18, 18> GetQ() {
    return Q;
}

void SetP(const Eigen::Matrix<double, 18, 18>& in_P) {
    P = in_P;
}
void SetQ(const Eigen::Matrix<double, 18, 18>& in_Q) {
    Q = in_Q;
}

private:
// 状态量
State state;
State last_state;

// 协方差矩阵
Eigen::Matrix<double, 18, 18> P;

// 系统噪声
Eigen::Matrix<double, 18, 18> Q;
};  // namespace eskf
}  // namespace eskf