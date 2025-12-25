// trajectory_publisher_node.cpp  
#include <boost/version.hpp>
#include <boost/filesystem.hpp>

#include <ros/ros.h>  
#include <geometry_msgs/Pose.h>  
#include <geometry_msgs/Point.h>  
#include <geometry_msgs/Quaternion.h>  
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
// #include <nav_msgs/Path.h>   
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>

#include <fstream>  
#include <sstream>  
#include <iostream>  
#include <map>
#include <limits>

#include <pcl_conversions/pcl_conversions.h>  
#include <pcl/point_types.h>

#include "../include/reliable_loc_demo_cpp/cnpy.h"
  

using namespace boost;

using FileVec = std::vector<std::string>;


#ifdef WIN32
#define DIR_INTERVAL '\\'
#else
#define DIR_INTERVAL '/'
#endif


// param
// std::string est_traj_file = "/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-22T23-13-10_cs_college_reliable_0.0005/xyzyaw_est.txt";
// std::string gt_traj_file = "/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-22T23-13-10_cs_college_reliable_0.0005/xyzyaw_gt.txt";
// std::string wls_bin_dir = "/home/ericxhzou/Data/benchmark_datasets/Self_Collected/For_Testing/whu_cs_college/helmet_submap/pointcloud_30m_5x5m";
// std::string mls_pc_file = "/home/ericxhzou/Data/temp/mls_pc_cs_college.txt";
// double g_offset_x = 533732.755952;
// double g_offset_y = 3380087.522021;
// double g_offset_z = 9.486584000000001;

std::string est_traj_file = "/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-26T21-05-22_zhongshan_park_reliable_0.0005/xyzyaw_est.txt";
std::string gt_traj_file = "/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-26T21-05-22_zhongshan_park_reliable_0.0005/xyzyaw_gt.txt";
std::string wls_bin_dir = "/home/ericxhzou/Data/benchmark_datasets/Self_Collected/For_Testing/wh_zhongshan_park/helmet_submap/pointcloud_30m_5x5m";
std::string mls_pc_file = "/home/ericxhzou/Data/temp/mls_pc_zhongshan_park.txt";
double g_offset_x = 794000.0;
double g_offset_y = 385000.0;
double g_offset_z = 30.0;

// std::string est_traj_file = "/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-23T20-28-27_info_campus_reliable_0.0005/xyzyaw_est.txt";
// std::string gt_traj_file = "/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-23T20-28-27_info_campus_reliable_0.0005/xyzyaw_gt.txt";
// std::string wls_bin_dir = "/home/ericxhzou/Data/benchmark_datasets/Self_Collected/For_Testing/whu_info_campus/helmet_submap/pointcloud_30m_5x5m";
// std::string mls_pc_file = "/home/ericxhzou/Data/temp/mls_pc_info_campus.txt";
// double g_offset_x = 534000.0;
// double g_offset_y = 3379000.0;
// double g_offset_z = 25.0;


void FormatPath(std::string &path) {
    size_t size = path.size();
    if (size == 0 || path[size - 1] != DIR_INTERVAL) {
        path.push_back(DIR_INTERVAL);
    }
}

std::string MakePath(const std::string &dir, const std::string &file) {
    std::string path = dir;
    FormatPath(path);
    return path + file;
}

template <class Container>
void TemplatedGetFiles(const std::string& dir,
    const std::string& ext,
    Container& container) {
    filesystem::path fullPath(filesystem::initial_path());
    fullPath = filesystem::system_complete(filesystem::path(dir));

    if (!filesystem::exists(fullPath) || !filesystem::is_directory(fullPath)) {
        return;
    }

    filesystem::directory_iterator end_iter;
    filesystem::directory_iterator file_itr(fullPath);
    for (; file_itr != end_iter; ++file_itr) {
        if (!filesystem::is_directory(*file_itr) &&
            (filesystem::extension(*file_itr) == ext || ext == "")) {
#if BOOST_VERSION > 104000
            std::string str = MakePath(dir, file_itr->path().filename().string());
#else
            std::string str = MakePath(dir, file_itr->path().filename());
#endif
            container.push_back(str);
        }
    }
}

void GetFiles(const std::string& dir, const std::string& ext, FileVec& files) {
    TemplatedGetFiles(dir, ext, files);
}

std::string GetExt(const std::string &file) {
    size_t pos = file.rfind('.');
    return (pos != std::string::npos) ? file.substr(pos) : std::string();
}

std::string GetName(const std::string &file) {
    boost::filesystem::path p(file);
#if BOOST_VERSION > 104000
    return p.filename().string();
#else
    return p.filename();
#endif
}

std::string GetNameWithoutExt(const std::string &file) {
    std::string name = GetName(file);
    std::string ext = GetExt(file);
    return name.substr(0, name.length() - ext.length());
}

bool CompareStamp(const std::pair<std::string, long>& a, const std::pair<std::string, long>& b) {
    return a.second < b.second;
}

bool LoadTraj(const std::string& filename, std::vector<bool>& is_reg,
              std::vector<geometry_msgs::Point>& point_msgs) {
    // 打开轨迹文件  
    std::ifstream trajectory_file(filename);  
    if (!trajectory_file.is_open()) {  
        ROS_ERROR("Unable to open trajectory file");  
        return false;  
    }
    std::string line;  
    while (std::getline(trajectory_file, line)) {  
        std::istringstream iss(line);  
        std::string loc_mode;
        double x, y, z, yaw;  
        if (!(iss >> loc_mode >> x >> y >> z >> yaw)) {  
            ROS_ERROR("Invalid line in trajectory file: %s", line.c_str());  
            continue;  
        }  

        if (loc_mode == "reg_loc")
            is_reg.push_back(true);
        else {
            is_reg.push_back(false);
        }
        
        // // 创建geometry_msgs/Pose消息  
        // geometry_msgs::Pose pose_msg;  
        // pose_msg.position.x = x;  
        // pose_msg.position.y = y;  
        // pose_msg.position.z = z; // 假设z为0，或者根据你的数据设置  
  
        // // 将yaw角转换为四元数  
        // tf2::Quaternion yaw_quat;  
        // yaw_quat.setRPY(0, 0, yaw);  
        // pose_msg.orientation.x = yaw_quat.x();  
        // pose_msg.orientation.y = yaw_quat.y();  
        // pose_msg.orientation.z = yaw_quat.z();  
        // pose_msg.orientation.w = yaw_quat.w();  

        // geometry_msgs::PoseStamped pose_stamped;
        // pose_stamped.header.stamp = ros::Time::now();
        // pose_stamped.header.frame_id = "map";
        // pose_stamped.pose = pose_msg;

        geometry_msgs::Point pt;
        pt.x = x;
        pt.y = y;
        pt.z = z;
        point_msgs.push_back(pt);
    }  
  
    // 关闭文件并结束ROS节点  
    trajectory_file.close();
    return true;
}

void SmoothTraj(const std::vector<geometry_msgs::Point>& in_point_msgs, std::vector<geometry_msgs::Point>& ou_point_msgs) {
    std::vector<geometry_msgs::Point> ou_msgs;
    ou_msgs.push_back(in_point_msgs[0]);
    geometry_msgs::Point pt_3rd;
    pt_3rd.x = (in_point_msgs[0].x + in_point_msgs[1].x + in_point_msgs[2].x) / 3;
    pt_3rd.y = (in_point_msgs[0].y + in_point_msgs[1].y + in_point_msgs[2].y) / 3;
    pt_3rd.z = (in_point_msgs[0].z + in_point_msgs[1].z + in_point_msgs[2].z) / 3;
    ou_msgs.push_back(pt_3rd);
    for (size_t i = 0; i + 4 < in_point_msgs.size(); ++i) {
        geometry_msgs::Point pt_5th;
        pt_5th.x = (in_point_msgs[i].x + in_point_msgs[i+1].x + in_point_msgs[i+2].x + in_point_msgs[i+3].x + in_point_msgs[i+4].x) / 5;
        pt_5th.y = (in_point_msgs[i].y + in_point_msgs[i+1].y + in_point_msgs[i+2].y + in_point_msgs[i+3].y + in_point_msgs[i+4].y) / 5;
        pt_5th.z = (in_point_msgs[i].z + in_point_msgs[i+1].z + in_point_msgs[i+2].z + in_point_msgs[i+3].z + in_point_msgs[i+4].z) / 5;
        ou_msgs.push_back(pt_5th);
    }
    int num_msg = in_point_msgs.size();
    pt_3rd.x = (in_point_msgs[num_msg-3].x + in_point_msgs[num_msg-2].x + in_point_msgs[num_msg-1].x) / 3;
    pt_3rd.y = (in_point_msgs[num_msg-3].y + in_point_msgs[num_msg-2].y + in_point_msgs[num_msg-1].y) / 3;
    pt_3rd.z = (in_point_msgs[num_msg-3].z + in_point_msgs[num_msg-2].z + in_point_msgs[num_msg-1].z) / 3;
    ou_msgs.push_back(pt_3rd);
    ou_msgs.push_back(in_point_msgs[num_msg-1]);
    ou_point_msgs = ou_msgs;
}

bool LoadWLS(const std::string& filename, sensor_msgs::PointCloud2& ros_cloud) {
    cnpy::NpyArray npy_array = cnpy::npy_load(filename);
    auto data = npy_array.as_vec<double>();
    if (data.empty() || data.size() % 3 != 0) {
        return false;
    }
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    int num_pt = data.size() / 3;
    for (int i = 0; i < num_pt; ++i) {
        double x = data[3*i] - g_offset_x;
        double y = data[3*i+1] - g_offset_y;
        double z = data[3*i+2] - g_offset_z;
        pcl_cloud.push_back(pcl::PointXYZ(x, y, z));
    }
    pcl::toROSMsg(pcl_cloud, ros_cloud);  
    ros_cloud.header.stamp = ros::Time::now();  
    ros_cloud.header.frame_id = "map"; // 根据你的实际情况设置frame_id
    return true;
}

bool LoadMLS(const std::string& filename, pcl::PointCloud<pcl::PointXYZI>& pcl_cloud) {
    std::ifstream ifs(filename); // 替换为你的文本文件路径  
    std::string line;  
    while (std::getline(ifs, line)) {  
        std::istringstream iss(line);  
        double x, y, z, intensity;  
        iss >> x >> y >> z >> intensity; 
        pcl::PointXYZI pt;
        pt.x = x-g_offset_x;
        pt.y = y-g_offset_y;
        pt.z = z-g_offset_z;
        pt.intensity = intensity;
        pcl_cloud.push_back(pt);
    }  
    ifs.close();
    return true;
}

struct Block {
    int id = 0;
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;

    Block(int id0=0) {
        id = id0;
    }
};

struct BlockInfo {
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    int nx = 0;
    int ny = 0;
    float block_size = 150.f;
};


void PC2Block(const pcl::PointCloud<pcl::PointXYZI>& pcl_cloud, std::map<int, Block>& blocks, BlockInfo& binfo) {
    // min max
    for (int i = 0; i < pcl_cloud.size(); ++i) {
        const auto& pt = pcl_cloud.at(i);
        if (pt.x < binfo.min_x)
            binfo.min_x = pt.x;
        if (pt.x > binfo.max_x)
            binfo.max_x = pt.x;
        if (pt.y < binfo.min_y)
            binfo.min_y = pt.y;
        if (pt.y > binfo.max_y)
            binfo.max_y = pt.y;
    }
    binfo.min_x -= std::numeric_limits<double>::epsilon() * 10;
    binfo.max_x += std::numeric_limits<double>::epsilon() * 10;
    binfo.min_y -= std::numeric_limits<double>::epsilon() * 10;
    binfo.max_y += std::numeric_limits<double>::epsilon() * 10;
    // blocks
    blocks.clear();
    binfo.nx = (binfo.max_x - binfo.min_x) / binfo.block_size + 1;
    binfo.ny = (binfo.max_y - binfo.min_y) / binfo.block_size + 1;
    for (int i = 0; i < pcl_cloud.size(); ++i) {
        const auto& pt = pcl_cloud.at(i);
        int id_x = (pt.x - binfo.min_x) / binfo.block_size;
        int id_y = (pt.y - binfo.min_y) / binfo.block_size;
        int id = id_x * binfo.ny + id_y;
        if (blocks.find(id) == blocks.end()) {
            blocks.insert(std::make_pair(id, Block(id)));
        }
        blocks[id].pcl_cloud.push_back(pt);
    }
    ROS_INFO("Obtain %d blocks from MLS Point Cloud", blocks.size());
}

void GetMLSFromBlocks(float x, float y, const std::map<int, Block>& blocks, const BlockInfo& binfo, sensor_msgs::PointCloud2& ros_cloud) {
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    int id_x = (x - binfo.min_x) / binfo.block_size;
    int id_y = (y - binfo.min_y) / binfo.block_size;
    for (int i_x = id_x - 4; i_x <= id_x + 4; ++i_x) {
        for (int i_y = id_y - 4; i_y <= id_y + 4; ++i_y) {
            int id = i_x * binfo.ny + i_y;
            if (blocks.find(id) == blocks.end()) {
                continue;
            }
            pcl_cloud += blocks.at(id).pcl_cloud;
        }
    }
    // 将PCL点云转换为ROS点云消息格式   
    pcl::toROSMsg(pcl_cloud, ros_cloud);  
    ros_cloud.header.stamp = ros::Time::now();  
    ros_cloud.header.frame_id = "map"; // 根据你的实际情况设置frame_id  
}


int main(int argc, char **argv)  
{  
    // 初始化ROS节点  
    ros::init(argc, argv, "pub_traj");  
    ros::NodeHandle nh;  
  
    // 创建发布者，发布轨迹消息  
    ros::Publisher trajectory_reg_pub = nh.advertise<visualization_msgs::Marker>("trajectory_reg", 10);
    ros::Publisher trajectory_pf_pub = nh.advertise<visualization_msgs::Marker>("trajectory_pf", 10);  
    ros::Publisher wls_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("wls_pc", 10);
    ros::Publisher mls_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("mls_pc", 1);
  
    // 打开est轨迹文件  
    std::vector<bool> est_is_reg;
    std::vector<geometry_msgs::Point> est_point_msgs;
    bool success = LoadTraj(est_traj_file, est_is_reg, est_point_msgs); 
    if (!success) {  
        ROS_ERROR("Unable to open EST trajectory file");  
        return -1;  
    }  

    // 打开gt轨迹文件  
    std::vector<bool> gt_is_reg;
    std::vector<geometry_msgs::Point> gt_point_msgs;
    success = LoadTraj(gt_traj_file, gt_is_reg, gt_point_msgs); 
    if (!success) {  
        ROS_ERROR("Unable to open GT trajectory file");  
        return -1;  
    } 
    SmoothTraj(gt_point_msgs, gt_point_msgs);
  
    // 读取并发布轨迹数据  
    visualization_msgs::Marker marker, marker_reg, marker_pf;
    marker.header.frame_id = "map"; // 设置参考坐标系
    marker.ns = "trajectory";  
    marker.id = 0;  
    marker.type = visualization_msgs::Marker::LINE_STRIP; // 使用线段条  
    marker.action = visualization_msgs::Marker::ADD;  
    marker.pose.position.x = 0.0;  
    marker.pose.position.y = 0.0;  
    marker.pose.position.z = 0.0;  
    marker.pose.orientation.x = 0.0;  
    marker.pose.orientation.y = 0.0;  
    marker.pose.orientation.z = 0.0;  
    marker.pose.orientation.w = 1.0;  
    marker.scale.x = 2.5; // 设置线宽  
    marker.color.a = 1.0; // 不透明度  

    marker_reg = marker;
    marker_reg.color.r = 1.0; // 红色  
    marker_reg.color.g = 0.0; // 绿色  
    marker_reg.color.b = 1.0; // 蓝色

    marker_pf = marker;
    marker_pf.color.r = 0.0; // 红色  
    marker_pf.color.g = 1.0; // 绿色  
    marker_pf.color.b = 1.0; // 蓝色

    tf::TransformBroadcaster br;
    tf::Transform transform;

    // bin files
    std::vector<std::string> bin_files;
    GetFiles(wls_bin_dir, ".bin", bin_files);
    if (bin_files.size() != est_point_msgs.size()) {  
        ROS_ERROR("Error!");  
        return -1;  
    }  
    std::vector<std::pair<std::string, long>> bin_stamps;
    for (const auto& bin : bin_files) {
        auto name = GetNameWithoutExt(bin);
        long stamp = std::atol(name.c_str());
        bin_stamps.push_back(std::make_pair(bin, stamp));
    }
    std::sort(bin_stamps.begin(), bin_stamps.end(), CompareStamp);

    // load mls pc
    pcl::PointCloud<pcl::PointXYZI> pcl_mls_pc;
    LoadMLS(mls_pc_file, pcl_mls_pc);
    std::map<int, Block> blocks;
    BlockInfo binfo;
    PC2Block(pcl_mls_pc, blocks, binfo);

    // 设置发布频率为10Hz  
    ros::Rate loop_rate(20);

    // 进入主循环  
    int count = 0;
    while (ros::ok())  
    {  
        // 发布轨迹消息  
        marker_reg.header.stamp = ros::Time::now();
        marker_pf.header.stamp = ros::Time::now();
        if (est_is_reg[count]) {
            marker_reg.points.push_back(est_point_msgs[count]);
            marker_pf.points.clear();
        } else {
            marker_pf.points.push_back(est_point_msgs[count]);
            marker_reg.points.clear();
        }
        trajectory_reg_pub.publish(marker_reg);
        trajectory_pf_pub.publish(marker_pf);

        // tf
        auto point_msg = gt_point_msgs[count];
        transform.setOrigin( tf::Vector3(point_msg.x, point_msg.y, point_msg.z));
        transform.setRotation( tf::Quaternion(0,0,0,1) );
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "lidar"));

        // load and publish pc
        auto bin = bin_stamps[count].first;
        sensor_msgs::PointCloud2 ros_wls_pc;
        LoadWLS(bin, ros_wls_pc);
        ros_wls_pc.header.stamp = ros::Time::now();
        wls_pc_pub.publish(ros_wls_pc);

        // publish mls pc
        if (count % 5 == 0) {
            sensor_msgs::PointCloud2 ros_mls_pc;
            GetMLSFromBlocks(point_msg.x, point_msg.y, blocks, binfo, ros_mls_pc);
            ros_mls_pc.header.stamp = ros::Time::now();
            mls_pc_pub.publish(ros_mls_pc);
        }

        ++count;
        if (count == est_point_msgs.size())
            count = 0;
        // 按照设定的频率休眠  
        loop_rate.sleep();  
    }
  
    return 0;  
}