#include "vio/imu.hh"

#include <random>  // 产生随机噪声
#include <sstream>
#include "vio/utilities.hh"

namespace vio {

IMU::IMU(SensorParams params) : param_(params) {
    gyro_bias_ = Eigen::Vector3d::Zero();
    acc_bias_ = Eigen::Vector3d::Zero();
}

// 根据时间产生位姿信息,然后对位姿求导获得速度,加速度等数据,这些数据不带噪声
MotionData IMU::MotionModel(double t) {
    MotionData data;
    // param
    float ellipse_x = 15;
    float ellipse_y = 20;
    float z = 1;    // z轴做sin运动
    float K1 = 10;  // z轴的正弦频率是x，y的k1倍
    float K = M_PI / 10;  // 20 * K = 2pi 　　由于我们采取的是时间是20s, 系数K控制yaw正好旋转一圈，运动一周

    // translation
    // twb:  body frame in world frame
    Eigen::Vector3d position(ellipse_x * cos(K * t) + 5, ellipse_y * sin(K * t) + 5, z * sin(K1 * K * t) + 5);
    Eigen::Vector3d dp(-K * ellipse_x * sin(K * t), K * ellipse_y * cos(K * t),
                       z * K1 * K * cos(K1 * K * t));  // position导数　in world frame
    double K2 = K * K;
    Eigen::Vector3d ddp(-K2 * ellipse_x * cos(K * t), -K2 * ellipse_y * sin(K * t),
                        -z * K1 * K1 * K2 * sin(K1 * K * t));  // position二阶导数

    // Rotation
    double k_roll = 0.1;
    double k_pitch = 0.2;
    Eigen::Vector3d eulerAngles(k_roll * cos(t), k_pitch * sin(t),
                                K * t);  // roll ~ [-0.2, 0.2], pitch ~ [-0.3, 0.3], yaw ~ [0,2pi]
    Eigen::Vector3d eulerAnglesRates(-k_roll * sin(t), k_pitch * cos(t), K);  // euler angles 的导数

    //    Eigen::Vector3d eulerAngles(0.0,0.0, K*t );   // roll ~ 0, pitch ~ 0, yaw ~ [0,2pi]
    //    Eigen::Vector3d eulerAnglesRates(0.,0. , K);      // euler angles 的导数

    Eigen::Matrix3d Rwb = euler2Rotation(eulerAngles);                                // body frame to world frame
    Eigen::Vector3d imu_gyro = eulerRates2bodyRates(eulerAngles) * eulerAnglesRates;  //  euler rates trans to body gyro

    Eigen::Vector3d gn(0, 0, -9.81);  //  gravity in navigation frame(ENU)   ENU (0,0,-9.81)  NED(0,0,9,81)
    Eigen::Vector3d imu_acc = Rwb.transpose() * (ddp - gn);  //  Rbw * Rwn * gn = gs

    data.imu_gyro = imu_gyro;
    data.imu_acc = imu_acc;
    data.Rwb = Rwb;
    data.twb = position;
    data.imu_velocity = dp;
    data.timestamp = t;
    return data;
}

void IMU::addIMUMotionDataWithNoise(MotionData& data) {
    // 噪声engine
    std::random_device rd;
    std::default_random_engine rd_engine(rd());
    // 噪声模型
    std::normal_distribution<double> noise(0.0, 1.0);

    // 模拟imu的测量角速度数据 = 角速度  + 随机噪声 + 零偏
    Eigen::Vector3d noise_gyro(noise(rd_engine), noise(rd_engine), noise(rd_engine));
    Eigen::Matrix3d gyro_sqrt_cov = param_.gyro_noise_sigma * Eigen::Matrix3d::Identity();
    data.imu_gyro = data.imu_gyro + gyro_sqrt_cov * noise_gyro / sqrt(param_.imu_timestep) + gyro_bias_;
    // 加速度
    Eigen::Vector3d noise_acc(noise(rd_engine), noise(rd_engine), noise(rd_engine));
    Eigen::Matrix3d acc_sqrt_cov = param_.acc_noise_sigma * Eigen::Matrix3d::Identity();
    data.imu_acc = data.imu_acc + acc_sqrt_cov * noise_acc / sqrt(param_.imu_timestep) + acc_bias_;
    // gyro零偏
    Eigen::Vector3d noise_gyro_bias(noise(rd_engine), noise(rd_engine), noise(rd_engine));
    gyro_bias_ += param_.gyro_bias_sigma * sqrt(param_.imu_timestep) * noise_gyro_bias;
    data.imu_gyro_bias = gyro_bias_;
    // acc零偏
    Eigen::Vector3d noise_acc_bias(noise(rd_engine), noise(rd_engine), noise(rd_engine));
    acc_bias_ += param_.acc_bias_sigma * sqrt(param_.imu_timestep) * noise_acc_bias;
    data.imu_acc_bias = acc_bias_;
}

/**
 * @brief imu运动学模型, 利用imu的数据来积分出imu的轨迹, 使用欧拉积分
 * @param data_path
 * @param traj_path
 */
void IMU::IntegrationIMU(std::string data_path, std::string traj_path) {
    std::ofstream f_writer;
    f_writer.open(traj_path);

    std::vector<MotionData> datas;
    auto load_datas = [&](std::string& path, std::vector<MotionData>& poses) {
        std::ifstream f_read;
        f_read.open(path);
        if (!f_read.is_open()) {
            std::cerr << " can't open LoadFeatures file " << std::endl;
            return;
        }
        while (!f_read.eof()) {
            std::string s;
            // 读取行
            std::getline(f_read, s);
            if (!s.empty()) {
                std::stringstream ss;
                ss << s;
                // 利用ss分割空格
                MotionData data;
                Eigen::Quaterniond q;
                Eigen::Vector3d p;
                Eigen::Vector3d gyro;
                Eigen::Vector3d acc;
                double time;
                ss >> time;
                ss >> q.w();
                ss >> q.x();
                ss >> q.y();
                ss >> q.z();
                ss >> p(0);
                ss >> p(1);
                ss >> p(2);
                ss >> gyro(0);
                ss >> gyro(1);
                ss >> gyro(2);
                ss >> acc(0);
                ss >> acc(1);
                ss >> acc(2);

                data.timestamp = time;
                data.imu_acc = acc;
                data.imu_gyro = gyro;
                data.Rwb = Eigen::Matrix3d(q);
                data.twb = p;
                poses.push_back(data);
            }
        }
    };
    // 加载数据
    load_datas(data_path, datas);

    // 开始积分
    double dt = param_.imu_timestep;
    Eigen::Vector3d P_wb = init_twb_;
    Eigen::Quaterniond Q_wb(init_Rwb_);
    Eigen::Vector3d V_wb = init_velocity_;
    Eigen::Vector3d g_(0.0, 0.0, -9.81);

    for (int i = 1; i < datas.size(); ++i) {
        MotionData imu_data = datas[i];
        MotionData imu_next_data = datas[i + 1];

        //------- 欧拉积分
        // dq = [1, 0.5 * delta_thetax,  0.5 * delta_thetay,  0.5 * delta_thetaz]
        // Eigen::Quaterniond dq;
        // Eigen::Vector3d d_theta_half = imu_data.imu_gyro * dt / 2;
        // dq.w() = 1;
        // dq.x() = d_theta_half.x();
        // dq.y() = d_theta_half.y();
        // dq.z() = d_theta_half.z();
        // dq.normalize();
        // // 计算世界坐标下的加速度
        // Eigen::Vector3d acc_w = Q_wb * imu_data.imu_acc + g_;

        // Q_wb = Q_wb * dq;
        // // P_wb
        // P_wb = P_wb + V_wb * dt + 0.5 * acc_w * dt * dt;
        // // V_wb
        // V_wb = V_wb + acc_w * dt;
        //------- 欧拉积分
        //---- 中值积分
        Eigen::Quaterniond dq;
        Eigen::Vector3d d_theta_half = 0.5 * (imu_data.imu_gyro + imu_next_data.imu_gyro) * dt / 2;
        dq.w() = 1;
        dq.x() = d_theta_half.x();
        dq.y() = d_theta_half.y();
        dq.z() = d_theta_half.z();
        dq.normalize();
        Eigen::Quaterniond Q_wb_tmp;
        Q_wb_tmp = Q_wb * dq;
        // 计算acc
        Eigen::Vector3d acc_w = (Q_wb * imu_data.imu_acc + Q_wb_tmp * imu_next_data.imu_acc) / 2.0 + g_;
        // 更新Q
        Q_wb = Q_wb * dq;
        // P_wb
        P_wb = P_wb + V_wb * dt + 0.5 * acc_w * dt * dt;
        // V_wb
        V_wb = V_wb + acc_w * dt;
        //---- 中值积分

        //　按着imu postion, imu quaternion , cam postion, cam quaternion 的格式存储，由于没有cam，所以imu存了两次
        f_writer << std::fixed << imu_data.timestamp << " " << Q_wb.w() << " " << Q_wb.x() << " " << Q_wb.y() << " "
                 << Q_wb.z() << " " << P_wb(0) << " " << P_wb(1) << " " << P_wb(2) << " " << Q_wb.w() << " " << Q_wb.x()
                 << " " << Q_wb.y() << " " << Q_wb.z() << " " << P_wb(0) << " " << P_wb(1) << " " << P_wb(2) << " "
                 << std::endl;
    }
}

}  // namespace vio