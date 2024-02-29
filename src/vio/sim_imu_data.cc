#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include "vio/imu.hh"
#include "vio/utilities.hh"

using namespace vio;

int main(int argc, char** argv) {
    ros::init(argc, argv, "sim_gener_imu_node");
    ros::NodeHandle nh;
    std::string imu_bag_path = "/home/kilox/kilox_ws/vio/imu_bag.bag";
    rosbag::Bag bag;
    bag.open(imu_bag_path, rosbag::bagmode::Write);

    double begin = ros::Time::now().toSec();
    std::cout << "Start generate imu data....." << std::endl;

    SensorParams params;
    IMU imu(params);

    std::vector<MotionData> imu_datas;
    std::vector<MotionData> imu_datas_noise;

    // const char symbols[4] = {'|', '/', '-', '\\'};
    for (double t = params.start_time; t < params.end_time; t += params.imu_timestep) {
        // if (static_cast<int>(t) % params.imu_hz == 0) {
        //     int i = static_cast<int>((t - params.start_time) / (params.end_time - params.start_time) * 100);
        //     printf("[#][%d%%][%c]\r", i, symbols[i % 4]);
        //     fflush(stdout);
        // }
        // creat imu data
        MotionData data = imu.MotionModel(t);
        imu_datas.push_back(data);
        MotionData data_noise = data;
        imu.addIMUMotionDataWithNoise(data_noise);
        imu_datas_noise.push_back(data_noise);
        // Eigen::Quaterniond q(data.Rwb);

        // ros::Time time_now(begin + t);
        // sensor_msgs::Imu imu_data;
        // imu_data.header.stamp = time_now;
        // imu_data.header.frame_id = "imu";

        // imu_data.orientation.x = q.x();
        // imu_data.orientation.y = q.y();
        // imu_data.orientation.z = q.z();
        // imu_data.orientation.w = q.w();

        // imu_data.linear_acceleration.x = data.imu_acc(0);
        // imu_data.linear_acceleration.y = data.imu_acc(1);
        // imu_data.linear_acceleration.z = data.imu_acc(2);

        // imu_data.angular_velocity.x = data.imu_gyro(0);
        // imu_data.angular_velocity.y = data.imu_gyro(1);
        // imu_data.angular_velocity.z = data.imu_gyro(2);

        // bag.write("\imu", time_now, imu_data);
    }
    imu.init_velocity_ = imu_datas[0].imu_velocity;
    imu.init_Rwb_ = imu_datas[0].Rwb;
    imu.init_twb_ = imu_datas[0].twb;

    auto save_pose = [&](const std::string& path, std::vector<MotionData> datas) {
        std::ofstream fopen;
        fopen.open(path);
        for (const MotionData data : datas) {
            double time = data.timestamp;
            Eigen::Quaterniond q(data.Rwb);
            Eigen::Vector3d t = data.twb;
            Eigen::Vector3d gyro = data.imu_gyro;
            Eigen::Vector3d acc = data.imu_acc;
            fopen << std::fixed << time << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << t(0)
                  << " " << t(1) << " " << t(2) << " " << gyro(0) << " " << gyro(1) << " " << gyro(2) << " " << acc(0)
                  << " " << acc(1) << " " << acc(2) << " " << std::endl;
        }
    };

    save_pose("/home/kilox/kilox_ws/vio/imu_data.txt", imu_datas);
    save_pose("/home/kilox/kilox_ws/vio/imu_data_noise.txt", imu_datas_noise);

    // fflush(stdout);
    // bag.close();
    std::cout << "Done, save imu data to" << imu_bag_path << std::endl;

    std::cout << "开始对imu数据积分,并保存imu的轨迹" << std::endl;
    imu.IntegrationIMU("/home/kilox/kilox_ws/vio/imu_data.txt", "/home/kilox/kilox_ws/vio/imu_traj.txt");
    imu.IntegrationIMU("/home/kilox/kilox_ws/vio/imu_data_noise.txt", "/home/kilox/kilox_ws/vio/imu_traj_noise.txt");

    return 0;
}