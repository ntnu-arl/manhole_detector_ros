#include <mutex>
#include <queue>
#include <memory.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/UInt8MultiArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgproc/imgproc.hpp"

#include <eigen3/Eigen/Dense>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

image_transport::Publisher depth_img_pub;

void cloudCallback(const sensor_msgs::PointCloud2 cloud)
{
	cv::Mat depth_img;
	pcl::PointCloud<pcl::PointXYZ> cloud_pcl;
	pcl::fromROSMsg(cloud, cloud_pcl);
	double max_range = 5.0;
	// std::cout << "cloud height: " << cloud.height << " cloud width: " << cloud.width << std::endl;
	// depth_img = cv::Mat::zeros(cloud.width, cloud.height, CV_32F);
	depth_img = cv::Mat::zeros(cloud.height, cloud.width, CV_32F);
	for(int i=0; i<cloud.height; ++i)
	{
		for(int j=0; j<cloud.width; ++j)
		{
			pcl::PointXYZ pt = cloud_pcl.at(j,i);
			if(!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
				depth_img.at<float>(i, j) = float(max_range);
				continue;
			}
			else {
				double r = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
				if(r > max_range) r = max_range;
				depth_img.at<float>(i, j) = float(r);
			}
		}
	}
	// std::cout << "1" << std::endl;
	cv::Mat flipped_image;
	cv::flip(depth_img, flipped_image, 0);
	cv::flip(flipped_image, flipped_image, 1);
	sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(cloud.header, "32FC1", flipped_image).toImageMsg();
	// std::cout << "2" << std::endl;
	depth_img_pub.publish(img_msg);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "depth_cloud_pub");
	ros::NodeHandle nh;

	image_transport::ImageTransport it(nh);
  	depth_img_pub = it.advertise("depth_img_out", 1);
	ros::Subscriber cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("input_cloud", 1, cloudCallback);

	ros::spin();

	return 0;
}