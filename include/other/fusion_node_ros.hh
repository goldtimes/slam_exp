#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include "gnss_imu_fusion.hh"

class FusionNodeRos {
   private:
    ros::Subscriber gnss_sub;
    ros::Subscriber imu_sub;
    ros::Publisher path_pub;
    ros::Publisher odom_pub;
    std::shared_ptr<GNSSImuFusion> fusion;

    int hz_mode = 0;

   public:
    FusionNodeRos(ros::NodeHandle &nh);
    ~FusionNodeRos();
    void imu_cb(const sensor_msgs::ImuPtr &msg);

    void normal_gnss(const sensor_msgs::NavSatFixPtr &msg);
    void pub_msgs(eskf::ESKF::State state, double time);
};