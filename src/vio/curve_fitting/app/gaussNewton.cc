#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    double ar = 1.0, br = 2.0, cr = 3.0;  // 真实的曲线参数
    // gaussNewton法的给定一个初始值
    double ae = 2.0, be = -1.0, ce = 5.0;  // 估计的曲线参数
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    // 随机数
    std::default_random_engine rd_engine;
    std::normal_distribution<double> noise(0.0, 1.0);

    std::vector<double> x_data;
    std::vector<double> y_data;

    // 产生数据
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.push_back(x);
        double tmp = exp(ar * x * x + br * x + cr) + noise(rd_engine);
        y_data.push_back(tmp);
    }

    int iterations = 100;
    double cost = 0;
    double lastCost = 0;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        // 定义Hessian矩阵和b矩阵
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();  // 因为待估计的参数是3,所以H 矩阵为3x3
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        cost = 0;

        for (int i = 0; i < N; ++i) {
            double xi = x_data[i];
            double y_observer = exp(ae * xi * xi + be * xi + ce);
            double error = y_data[i] - y_observer;
            // 求雅克比矩阵, de / da, de / db, de / dc
            Eigen::Vector3d J;
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);
            J[2] = -exp(ae * xi * xi + be * xi + ce);
            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error;
        }
        //  求增量 H*x=b
        Eigen::Vector3d dx = H.ldlt().solve(b);
        if (std::isnan(dx[0])) {
            std::cout << "result is nan!, break while" << std::endl;
            break;
        }
        // 这次的误差大于下次的误差,迭代有问题
        if (iter > 0 && cost >= lastCost) {
            std::cout << "cost:" << cost << ">= last cost:" << lastCost << ",break." << std::endl;
            break;
        }
        // 更新状态量
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        // 记录cost
        lastCost = cost;
        // 打印状态量,观察变化, 当dx很小的时候,也可以break
        std::cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae
                  << "," << be << "," << ce << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start_time);
    std::cout << "solve time cost=" << time_used.count() << " seconds" << std::endl;
    std::cout << "solve estimated abc = " << ae << ", " << be << ", " << ce << std::endl;
    return 0;
}