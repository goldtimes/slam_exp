#include <ceres/ceres.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : x_(x), y_(y) {
    }

    template <typename T>
    /**
     * @brief 计算残差
     * @param abc 待优化的参数
     * @param residual 残差
     */
    bool operator()(const T* const abc, T* residual) const {
        residual[0] = T(y_) - ceres::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
        return true;
    }

    const double x_;
    const double y_;
};

int main(int argc, char** argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;

    std::default_random_engine rd_engine;
    std::normal_distribution<double> noise(0.0, 1.0);

    std::vector<double> x_data;
    std::vector<double> y_data;
    // 构造数据
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        double y = exp(ar * x * x + br * x + cr) + noise(rd_engine);
        x_data.push_back(x);
        y_data.push_back(y);
    }
    double abc[3] = {ae, be, ce};

    ceres::Problem problem;
    // 添加误差项
    for (int i = 0; i < N; ++i) {
        problem.AddResidualBlock(
            // 创建自动求导的costFunction, 类型CURVE_FITTING_COST, 残差维度,待优化的变量维度
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),
            nullptr, abc);
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // Hx=b的求解方式
    options.minimizer_progress_to_stdout = true;                // 输出到cout

    ceres::Solver::Summary summary;  // 优化信息
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // 开始优化
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

    std::cout << "solve time cost = " << time_used.count() << " seconds." << std::endl;

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "estimated a,b,c=";
    for (const auto& value : abc) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    return 0;
}