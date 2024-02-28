#include "sac_ia.hh"

int main(int argc, char **argv) {
    ros::init(argc, argv, "sac_ia");
    ros::NodeHandle nh;
    std::string map_dir = "/home/kilox/kilox_ws/wutongdao/meta_maps";
    int floor = 1;
    auto relocalization = TemplateAligment(map_dir, floor);
    std::string submap_path = "/home/kilox/kilox_ws/shinei_submap.pcd";
    PointCloud::Ptr submap;
    submap.reset(new PointCloud);
    pcl::io::loadPCDFile(submap_path, *submap);
    Eigen::Matrix4d tf;
    pcl::Registration<pcl::PointXYZ, pcl::PointXYZ, float>::Matrix4 matrix4transform;
    float score;
    relocalization.findBestAligment(submap, tf, matrix4transform, score);
    ros::spin();
    return 0;
}