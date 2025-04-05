#include <mutex>
#include <queue>
#include <memory.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/UInt8MultiArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <std_srvs/Trigger.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/pca.h>
#include <pcl/filters/crop_box.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgproc/imgproc.hpp"

#include <eigen3/Eigen/Dense>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include "planner_msgs/OpeningDetection.h"
#include "planner_msgs/MultipleOpeningDetections.h"

struct ManholeDetection
{
	int id;
	bool seen_sufficiently = false;
	std::vector<Eigen::Vector4d> detections;
	std::vector<int> detection_buffer;
	Eigen::Vector3d mean_pos;
	double mean_yaw;
};

class ManholeDetector
{
private:
	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_;

	ros::Subscriber sub_depth_img_;
	ros::Subscriber sub_cloud_;
	ros::Subscriber sub_odom_;

	ros::Publisher pub_contour_pcl_;
	ros::Publisher pub_contour_centroid_;
	ros::Publisher pub_detections_;
	ros::Publisher pub_detections_vis_;
	ros::Publisher pub_detection_odom_;
	ros::Publisher pub_processed_cloud_;
	image_transport::Publisher pub_contour_img_, pub_cropped_img_;

	ros::ServiceServer srv_reset_detection_poses_;

	std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
	tf2_ros::Buffer tf_buffer_;

	std::vector<ManholeDetection> manhole_detections_;

	bool first_detection_ = false;
	Eigen::Vector3d first_mh_location_;
	geometry_msgs::PoseArray detection_attempt_poses_;

	// Point cloud buffer
	std::vector<sensor_msgs::PointCloud2> pcl_cloud_buffer_;
	std::mutex pcl_msg_queue_mutex_;
	std::vector<std::vector<pcl::PointXYZI>> depth_img_to_pcl_map_;

	int manhole_count_ = 0;

	// Parameters
	Eigen::Vector3d manhole_bounding_box_size_min_, manhole_bounding_box_size_max_;
	int canny_low_thres_;
	int slack_;
	int col_offset_, row_offset_;
	double cluster_radius_;
	int buffer_size_;
	int min_detection_to_be_permenant_;
	double min_manhole_robot_distance_;
	double max_normal_pitch_;
	Eigen::Vector2i indices_to_crop_;
	bool crop_out_rows_;
	Eigen::Vector3d mh_centroid_bbox_;
	int max_points_inside_mh_;
	bool use_px_offsets_;
	int num_laser_beams_;

	std::vector<int> px_offsets_ = {23,
            3,
            23,
            3,
            23,
            4,
            22,
            4,
            22,
            4,
            22,
            4,
            22,
            5,
            22,
            5,
            22,
            5,
            22,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            5,
            21,
            4,
            21,
            4,
            21,
            4,
            21,
            4,
            21,
            3,
            21,
            3,
            21,
            3,
            21,
            2,
            21,
            2,
            21,
            1,
            21,
            0};

	void truncateYaw(double &yaw)
	{
		if (yaw > M_PI)
		{
			yaw -= 2 * M_PI;
		}
		if (yaw < -M_PI)
		{
			yaw += 2 * M_PI;
		}
	}
	
	void correctDetectionYaw(double &yaw)
	{
		if (yaw > M_PI)
		{
			yaw -= 2 * M_PI;
		}
		if (yaw < -M_PI)
		{
			yaw += 2 * M_PI;
		}

		if (yaw > M_PI / 2.0)
		{
			yaw -= M_PI;
		}
		if (yaw < -M_PI / 2.0)
		{
			yaw += M_PI;
		}
	}

	void correctDetectionYaw(double &yaw, double mean_yaw)
	{
		if (yaw > M_PI)
		{
			yaw -= 2 * M_PI;
		}
		if (yaw < -M_PI)
		{
			yaw += 2 * M_PI;
		}


		double d_yaw = yaw - mean_yaw;
		if (d_yaw > M_PI)
		{
			d_yaw -= 2 * M_PI;
		}
		if (d_yaw < -M_PI)
		{
			d_yaw += 2 * M_PI;
		}

		if (d_yaw > M_PI / 2.0)
		{
			d_yaw -= M_PI;
		}
		if (d_yaw < -M_PI / 2.0)
		{
			d_yaw += M_PI;
		}

		yaw = mean_yaw + d_yaw;
		if (yaw > M_PI)
		{
			yaw -= 2 * M_PI;
		}
		if (yaw < -M_PI)
		{
			yaw += 2 * M_PI;
		}
	}

	Eigen::Vector3d manhole_pos;
	nav_msgs::Odometry current_odom_;

public:
	ManholeDetector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);

	// Callbacks
	void depthImgCallback(const sensor_msgs::ImageConstPtr &img);
	void pclCallback(const sensor_msgs::PointCloud2ConstPtr &pcl_msg);
	void odomCallback(const nav_msgs::Odometry &odom);
	bool resetDetectionPoseList(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) 
	{
		detection_attempt_poses_.poses.clear();
		return true;
	}

	// Core Detection
	void detectClosedContours(cv::Mat &img, cv::Mat &out_img, std::vector<std::vector<cv::Point>> &closed_contours);
	void filterContours(std::vector<std::vector<cv::Point>> &closed_contours, pcl::PointCloud<pcl::PointXYZ> &cloud, cv::Mat image);
	void filterContours(cv::Mat &contours_img, pcl::PointCloud<pcl::PointXYZ> &cloud);
	void reindexContours(std::vector<std::vector<cv::Point>> &closed_contours, std::vector<std::vector<cv::Point>> &out_contours);

	// Utils
	void findCorrespondingPointCloud(const sensor_msgs::ImageConstPtr &img, pcl::PointCloud<pcl::PointXYZ> &cloud);
	double getDistance(pcl::PointXYZ pt) { return std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z); }
	void drawContours(cv::Mat &image, std::vector<std::vector<cv::Point>> closed_contours);

	bool getStableManholes(std::vector<ManholeDetection> &detections);
	void reEvaluate(int id);

	// Params
	bool loadParams();
};