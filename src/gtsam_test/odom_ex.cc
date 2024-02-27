#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>  // 优化器
#include <gtsam/nonlinear/Marginals.h>                    // 边缘化协方差
#include <gtsam/nonlinear/NonlinearFactorGraph.h>         // 非线性因子图
#include <gtsam/slam/BetweenFactor.h>                     // 二元因子
#include <gtsam/slam/PriorFactor.h>                       // 一元因子

int main(int argc, char** argv) {
    /**
     * 这个例子需要结合gtsam的官网 第一个例子,定义了3个变量,1个一元因子和二个二元因子
     */
    // 创建一个因子图
    gtsam::NonlinearFactorGraph graphFactor;

    // 先验均值
    gtsam::Pose2 priorMean(0.0, 0.0, 0.0);
    // 先验的标准差
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
    // 构造了先验因子
    graphFactor.add(gtsam::PriorFactor<gtsam::Pose2>(1, priorMean, priorNoise));

    // 添加里程计因子
    gtsam::Pose2 odom(2.0, 0.0, 0.0);
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.2, 0.2, 0.1));

    graphFactor.add(gtsam::BetweenFactor<gtsam::Pose2>(1, 2, odom, odom_noise));
    graphFactor.add(gtsam::BetweenFactor<gtsam::Pose2>(2, 3, odom, odom_noise));

    graphFactor.print("\nFactor Graph: \n");

    // 创建变量以及初始值
    gtsam::Values initial;
    initial.insert(1, gtsam::Pose2(0.5, 0, 0.2));
    initial.insert(2, gtsam::Pose2(2.3, 0.1, -0.2));
    initial.insert(3, gtsam::Pose2(4.1, 0.1, 0.1));
    initial.print("\n Initial Estimate:\n");

    gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graphFactor, initial).optimize();
    result.print("Final result: \n");

    gtsam::Marginals marginals(graphFactor, result);
    std::cout << "x1 covariance:\n" << marginals.marginalCovariance(1) << std::endl;
    std::cout << "x2 covariance:\n" << marginals.marginalCovariance(2) << std::endl;
    std::cout << "x3 covariance:\n" << marginals.marginalCovariance(3) << std::endl;

    return 0;
}