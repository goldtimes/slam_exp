#pragma once
#include "vo_slam/common_include.hh"

/**
 * @brief 配置类
 * 使用SetParameterFile确定配置文件
 * 然后用Get得到对应值
 */
namespace vo {
class Config {
   private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;
    Config() {
    }

   public:
    ~Config();

    static bool SetParameterFile(const std::string& filename);

    template <typename T>
    static T Get(const std::string& key) {
        return T(Config::Config_->file_[key]);
    }
};
}  // namespace vo