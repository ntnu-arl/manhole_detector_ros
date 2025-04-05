#include "manhole_detector/manhole_detector.hpp"

ManholeDetector::ManholeDetector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private)
	: nh_(nh), nh_private_(nh_private)
{
	// px_offsets_ = std::vector<int>(64,0);
	
	if (!loadParams())
	{
		ROS_ERROR("Cannot load params.");
		ros::shutdown();
	}

	sub_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>("pcl_in", 10, &ManholeDetector::pclCallback, this);
	sub_depth_img_ = nh_.subscribe<sensor_msgs::Image>("depth_img_in", 10, &ManholeDetector::depthImgCallback, this);
	sub_odom_ = nh_.subscribe("/msf_core/odometry", 10, &ManholeDetector::odomCallback, this);

	std::cout << "Subs created" << std::endl;

	image_transport::ImageTransport it(nh_);
	pub_contour_img_ = it.advertise("contour_img", 1);
	pub_cropped_img_ = it.advertise("cropped_img", 1);

	std::cout << "Img pub created" << std::endl;

	pub_contour_pcl_ = nh_.advertise<sensor_msgs::PointCloud2>("contour_pcl", 1);
	pub_contour_centroid_ = nh_.advertise<geometry_msgs::PoseArray>("contour_centroids", 1);
	pub_detection_odom_ = nh_.advertise<nav_msgs::Odometry>("detected_locations", 1);

	std::cout << "Contour pub created" << std::endl;

	pub_detections_vis_ = nh_.advertise<geometry_msgs::PoseArray>("stable_detections_vis", 10);
	pub_detections_ = nh_.advertise<planner_msgs::MultipleManholeDetections>("stable_detections", 10);
	// pub_detections_ = nh_.advertise<nav_msgs::Path>("stable_detections", 10);

	std::cout << "Detections pub created" << std::endl;

	pub_processed_cloud_ = nh_.advertise<geometry_msgs::PoseArray>("detection_attempt", 10);

	srv_reset_detection_poses_ = nh_.advertiseService("reset_detection_poses",
                           &ManholeDetector::resetDetectionPoseList, this);

	tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);

}

void ManholeDetector::findCorrespondingPointCloud(const sensor_msgs::ImageConstPtr &img, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
	int closest_pcl = 0;
	int64_t min_time_diff = std::numeric_limits<int64_t>::max();
	for (int i = 0; i < pcl_cloud_buffer_.size(); ++i)
	{
		int64_t diff = std::abs(((int64_t)pcl_cloud_buffer_[i].header.stamp.toNSec() - (int64_t)img->header.stamp.toNSec()));
		if (diff < min_time_diff)
		{
			min_time_diff = diff;
			closest_pcl = i;
		}
	}

	pcl::fromROSMsg(pcl_cloud_buffer_[closest_pcl], cloud);
}

void ManholeDetector::pclCallback(const sensor_msgs::PointCloud2ConstPtr &pcl_msg)
{
	std::lock_guard<std::mutex> guard(pcl_msg_queue_mutex_);
	pcl_cloud_buffer_.insert(pcl_cloud_buffer_.begin(), *pcl_msg);
	if (pcl_cloud_buffer_.size() > 10)
	{
		pcl_cloud_buffer_.erase(pcl_cloud_buffer_.end());
	}
}

void ManholeDetector::depthImgCallback(const sensor_msgs::ImageConstPtr &img)
{
	auto t1 = std::chrono::high_resolution_clock::now();
  	auto t2 = t1;
	depth_img_to_pcl_map_.clear();
	pcl::PointCloud<pcl::PointXYZ> corresponding_pcl;
	if(pcl_cloud_buffer_.empty())
	{
		std::cout << "No point cloud received" << std::endl;
		return;
	}
	findCorrespondingPointCloud(img, corresponding_pcl);
	// std::cout << "Found corresponding cloud" << std::endl;

	cv::Mat img_in;
	cv::Mat depth_img;

	img_in = cv_bridge::toCvCopy(img, "mono16")->image;

	cv::Mat img_div2(img_in.rows, img_in.cols, CV_16U);
	img_div2 = img_in;

	double min = 0.0;
	double max = 99999.0;
	cv::minMaxLoc(img_div2, &min, &max);
	if (min == max)
	{
		min = 0;
		max = 2;
	}
	cv::Mat(img_div2 - min).convertTo(depth_img, CV_8U, 255.0 / (max - min));
	cv::Mat temp;

	cv::Mat contours_img(img_div2.rows, img_div2.cols, CV_8UC3), cropped_contours_img(depth_img.rows - (indices_to_crop_(1) - indices_to_crop_(0) + 1), depth_img.cols, CV_8UC3);
	std::vector<std::vector<cv::Point>> detected_contours;

	cv::Mat cropped_img(depth_img.rows - (indices_to_crop_(1) - indices_to_crop_(0) + 1), depth_img.cols, depth_img.type()); // extreme indices in indices_to_crop_ are to be removed as well
	if(crop_out_rows_)
	{
		cv::Mat i1, i2;
		depth_img(cv::Range(0,indices_to_crop_(0)), cv::Range(0,depth_img.cols)).copyTo(i1);
		depth_img(cv::Range(indices_to_crop_(1) + 1, depth_img.rows), cv::Range(0,depth_img.cols)).copyTo(i2); // extreme indices in indices_to_crop_ are to be removed as well
		cv::vconcat(i1, i2, cropped_img);

		cv::Mat frame_mask;
		cv::inRange(depth_img, 0, 0, frame_mask);
		if (pub_cropped_img_.getNumSubscribers() > 0)
		{
			sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(img->header, "mono8", frame_mask).toImageMsg();
			pub_cropped_img_.publish(img_msg);
		}
	}


	auto t1_c = std::chrono::high_resolution_clock::now();
  	auto t2_c = t1_c;

	if(crop_out_rows_)
	{
		detectClosedContours(cropped_img, cropped_contours_img, detected_contours);
		cv::Mat reindexed_contour_img;
		cv::cvtColor(depth_img, reindexed_contour_img, cv::COLOR_GRAY2BGR);
		std::vector<std::vector<cv::Point>> reindexed_contours;
		reindexContours(detected_contours, reindexed_contours);
		drawContours(reindexed_contour_img, reindexed_contours);

		filterContours(reindexed_contours, corresponding_pcl, depth_img);
		if (pub_contour_img_.getNumSubscribers() > 0)
		{
			std_msgs::Header new_header;
			new_header.frame_id = img->header.frame_id;
			new_header.stamp = img->header.stamp;
			sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(new_header, "bgr8", reindexed_contour_img).toImageMsg();
			pub_contour_img_.publish(img_msg);
		}
	}
	else
	{
		detected_contours.clear();
		detectClosedContours(depth_img, contours_img, detected_contours);
		t2_c = std::chrono::high_resolution_clock::now();
		double dt_c = std::chrono::duration<double, std::milli>(t2_c - t1_c).count();

		filterContours(detected_contours, corresponding_pcl, depth_img);

		t2 = std::chrono::high_resolution_clock::now();
		double dt = std::chrono::duration<double, std::milli>(t2 - t1).count();

		if (pub_contour_img_.getNumSubscribers() > 0)
		{
			std_msgs::Header new_header;
			new_header.frame_id = img->header.frame_id;
			new_header.stamp = img->header.stamp;
			sensor_msgs::Image::Ptr img_msg = cv_bridge::CvImage(new_header, "bgr8", contours_img).toImageMsg();
			pub_contour_img_.publish(img_msg);
		}
	}

	
}

void ManholeDetector::drawContours(cv::Mat &image, std::vector<std::vector<cv::Point>> closed_contours)
{
	for (int i = 0; i < closed_contours.size(); ++i)
	{
		cv::Vec3b color(((double)i / closed_contours.size()) * 255, 255, (1 - (double)i / closed_contours.size()) * 255);
		for (auto &p_orig : closed_contours[i])
		{
			image.at<cv::Vec3b>(p_orig) = color;
		}
	}
}

void ManholeDetector::detectClosedContours(cv::Mat &img, cv::Mat &out_img, std::vector<std::vector<cv::Point>> &closed_contours)
{
	cv::Mat edge_extracted_img(img.rows, img.cols, CV_8U);

	int low_thres = canny_low_thres_;
	int ratio = 3;
	int kernel_size = 3;

	cv::Mat temp;
	temp.create(img.size(), img.type());
	temp = img;
	cv::Canny(temp, edge_extracted_img, low_thres, low_thres * ratio, kernel_size);

	cv::cvtColor(img, out_img, cv::COLOR_GRAY2BGR);
	cv::Mat closed_contour_img;
	std::vector<std::vector<cv::Point>> contours; // std::Vector for storing contour
	closed_contours.clear();
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(edge_extracted_img, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

	
	for (int i = 0; i < contours.size(); i = hierarchy[i][0]) // iterate through each contour.
	{
		cv::Rect r = cv::boundingRect(contours[i]);
		if (cv::contourArea(contours[i]) > cv::arcLength(contours[i], true))
		{
			closed_contours.push_back(contours[i]);
		}
	}
	
	cv::Mat contours_only, contours_only_dilated;
	contours_only.create(img.size(), img.type());
	contours_only.setTo(cv::Scalar(0));
	cv::cvtColor(contours_only, contours_only, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < closed_contours.size(); ++i)
	{
		cv::Scalar color = cv::Scalar(((double)i / closed_contours.size()) * 255, 255, (1 - (double)i / closed_contours.size()) * 255);

		cv::drawContours(out_img, closed_contours, i, color, 1, cv::LINE_4, hierarchy, 0);
		cv::drawContours(contours_only, closed_contours, i, color, 1, cv::LINE_4, hierarchy, 0);
	}

}

void ManholeDetector::reindexContours(std::vector<std::vector<cv::Point>> &closed_contours, std::vector<std::vector<cv::Point>> &out_contours)
{
	int index_offset = indices_to_crop_(1) - indices_to_crop_(0) + 1;
	out_contours.clear();
	for (auto &contour : closed_contours)
	{
		std::vector<cv::Point> reindexed_contour;
		for (auto &p_orig : contour)
		{
			cv::Point p_new = p_orig;
			if(p_orig.y >= indices_to_crop_(0))
			{
				p_new.y += index_offset;
			}
			reindexed_contour.push_back(p_new);
		}
		out_contours.push_back(reindexed_contour);
	}
}

void ManholeDetector::filterContours(cv::Mat &contours_img, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr contour_pcl(new pcl::PointCloud<pcl::PointXYZ>);
	cv::Mat contours_img_grey;
	cv::cvtColor(contours_img, contours_img_grey, cv::COLOR_BGR2GRAY);

	int y_offset = 10;

	for (int x = 0; x < contours_img.rows; ++x)
	{
		for (int y = 0; y < contours_img.cols; ++y)
		{
			uchar pxl = contours_img_grey.at<uchar>(x, y);
			// if (pxl > 0)
			// 	std::cout << pxl << " \n";
			if (pxl > 250)
			{
				pcl::PointXYZ pi = cloud.at(std::max(y - y_offset, 0), x);
				contour_pcl->points.push_back(pi);
			}
		}
	}

	sensor_msgs::PointCloud2 cloud_msg;
	contour_pcl->header.frame_id = cloud.header.frame_id;
	pcl::toROSMsg(*contour_pcl, cloud_msg);
	cloud_msg.header.frame_id = cloud.header.frame_id;
	pub_contour_pcl_.publish(cloud_msg);
}

void ManholeDetector::filterContours(std::vector<std::vector<cv::Point>> &closed_contours, pcl::PointCloud<pcl::PointXYZ> &cloud, cv::Mat image)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr all_contours_pcl(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr all_contours_pcl_intensity(new pcl::PointCloud<pcl::PointXYZI>);

	int counter = 0;

	geometry_msgs::PoseArray centroids;
	// centroids.header.frame_id = cloud.header.frame_id;
	centroids.header.frame_id = "world";
	

	int num_manholes = 0;

	std::vector<std::pair<Eigen::Vector3d, double>> detections;

	cv::Mat test_img = cv::Mat::zeros(cloud.height, cloud.width, CV_8U);

	double total_plane_seg_time = 0.0;
	
	sensor_msgs::PointCloud2 in_cloud;
	pcl::toROSMsg(cloud, in_cloud);

	double dt_t;
	auto t1_t = std::chrono::high_resolution_clock::now();
	auto t2_t = t1_t;
	geometry_msgs::TransformStamped T_W_sensor;
	bool found = true;
	try
	{
		T_W_sensor = tf_buffer_.lookupTransform("world", cloud.header.frame_id,
												in_cloud.header.stamp, ros::Duration(0.1));
	}
	catch (tf2::TransformException &ex)
	{
		// ROS_WARN("%s", ex.what());
		found = false;
		// ros::Duration(1.0).sleep();
	}
	t2_t = std::chrono::high_resolution_clock::now();
	dt_t = std::chrono::duration<double, std::milli>(t2_t - t1_t).count();
	double transform_time = dt_t;
	// std::cout << "Transform time: " << transform_time << "ms" << std::endl;
	
	if(!found) return;

	


	for (auto &contour : closed_contours)
	{
		std::vector<Eigen::Vector4d> contour_pts;
		Eigen::Vector3d contour_centroid;
		contour_centroid = Eigen::Vector3d::Zero();
		Eigen::Vector2i contour_centroid_2d;
		contour_centroid_2d = Eigen::Vector2i::Zero();

		pcl::PointCloud<pcl::PointXYZ>::Ptr contour_pcl(new pcl::PointCloud<pcl::PointXYZ>);		

		for (auto &p_orig : contour)
		{
			// Eigen::Vector2d new_pix = (Eigen::Vector2d(p.x, p.y) - contour_centroid_2d) *
			cv::Point p = p_orig;

			int p_sel_x, p_sel_y;
			int ring = p.y;
			int col = (p.x + cloud.width - px_offsets_[ring]) % cloud.width;
			pcl::PointXYZ pi_0 = cloud.points[ring * cloud.width + col];

			int new_x, new_y;
			pcl::PointXYZ pi = pi_0;
			float min_dist = getDistance(pi_0);
			if(min_dist <= min_manhole_robot_distance_)
			{
				continue;
			}
			new_y = std::min(p.y + slack_, (int)cloud.height - 1);
			new_x = (p.x + cloud.width - px_offsets_[new_y]) % cloud.width; 

			pcl::PointXYZ pi_t = cloud.points[new_y * cloud.width + new_x];
			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = p.x; 
				p_sel_y = std::min(p.y + slack_, (int)cloud.height - 1);
			}
			new_y =  std::max(p.y - slack_, 0);
			new_x = (p.x + cloud.width - px_offsets_[new_y]) % cloud.width;
			pi_t = cloud.points[new_y * cloud.width + new_x];

			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = p.x;
				p_sel_y =  std::max(p.y - slack_, 0);
				
			}
			new_y =  p.y;
			new_x = (std::min(p.x + slack_, (int)cloud.width - 1) + cloud.width - px_offsets_[new_y]) % cloud.width;
			pi_t = cloud.points[new_y * cloud.width + new_x];

			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = std::min(p.x + slack_, (int)cloud.width - 1);
				p_sel_y =  p.y;
			}
			new_y = p.y;
			new_x = (std::max(p.x - slack_, 0) + cloud.width - px_offsets_[new_y]) % cloud.width;
			pi_t = cloud.points[new_y * cloud.width + new_x];

			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = std::max(p.x - slack_, 0);
				p_sel_y = p.y;
			}
			new_y = std::max(p.y - slack_, 0);
			new_x = (std::max(p.x - slack_, 0) + cloud.width - px_offsets_[new_y]) % cloud.width;
			pi_t = cloud.points[new_y * cloud.width + new_x];

			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = std::max(p.x - slack_, 0);
				p_sel_y = std::max(p.y - slack_, 0);
			}
			new_y = std::max(p.y - slack_, 0);
			new_x = (std::min(p.x + slack_, (int)cloud.width - 1) + cloud.width - px_offsets_[new_y]) % cloud.width;
			pi_t = cloud.points[new_y * cloud.width + new_x];

			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = std::min(p.x + slack_, (int)cloud.width - 1);
				p_sel_y = std::max(p.y - slack_, 0);
			}
			new_y = std::min(p.y + slack_, (int)cloud.height - 1);
			new_x = (std::max(p.x - slack_, 0) + cloud.width - px_offsets_[new_y]) % cloud.width;
			pi_t = cloud.points[new_y * cloud.width + new_x];

			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = std::max(p.x - slack_, 0);
				p_sel_y = std::min(p.y + slack_, (int)cloud.height - 1);
			}
			new_y = std::min(p.y + slack_, (int)cloud.height - 1);
			new_x = (std::min(p.x + slack_, (int)cloud.width - 1) + cloud.width - px_offsets_[new_y]) % cloud.width;
			pi_t = cloud.points[new_y * cloud.width + new_x];

			if (getDistance(pi_t) < min_dist && getDistance(pi_t) > min_manhole_robot_distance_ && std::abs(getDistance(pi_t) - min_dist) > 0.4)
			{
				pi = pi_t;
				min_dist = getDistance(pi_t);
				p_sel_x = std::min(p.x + slack_, (int)cloud.width - 1);
				p_sel_y = std::min(p.y + slack_, (int)cloud.height - 1);
			}

			Eigen::Vector4d pv(pi.x, pi.y, pi.z, getDistance(pi));
			contour_pts.push_back(pv);
			// contour_centroid += pv.head(3);
			contour_pcl->points.push_back(pi);
			all_contours_pcl->points.push_back(pi);

			test_img.at<uchar>(p.y, p.x) = 255;
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_contour_pcl(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto p : contour_pcl->points)
		{
			double r = p.x * p.x + p.y * p.y + p.z * p.z;
			if ((r >= 0.5))
				filtered_contour_pcl->push_back(p);
		}

		// std::cout << "Contour pcl size: " << contour_pcl->points.size() << std::endl;

		if(contour_pcl->points.size() <= 3)
		{
			continue;
		}

		// Plane fitting
		double dt;
		auto t1 = std::chrono::high_resolution_clock::now();
  		auto t2 = t1;
		int debug_count = 0;
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(0.1);
		seg.setInputCloud(contour_pcl);
		bool success = true;
		try
		{
			seg.segment(*inliers, *coefficients);
		}
		catch(...)
		{
			success = false;
		}
		t2 = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration<double, std::milli>(t2 - t1).count();
		total_plane_seg_time += dt;

		if(!success) 
		{
			continue;
		}

		if(inliers->indices.empty())
		{
			continue;
		}

		Eigen::Vector3d plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);

		Eigen::Quaterniond quat = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(1.0, 0.0, 0.0), plane_normal);

		geometry_msgs::PoseStamped plane_normal_msg, plane_normal_msg_W;
		plane_normal_msg.pose.position.x = plane_normal.x();
		plane_normal_msg.pose.position.y = plane_normal.y();
		plane_normal_msg.pose.position.z = plane_normal.z();
		plane_normal_msg.pose.orientation.x = 0.0;
		plane_normal_msg.pose.orientation.y = 0.0;
		plane_normal_msg.pose.orientation.z = 0.0;
		plane_normal_msg.pose.orientation.w = 1.0;
		geometry_msgs::TransformStamped R_W_sensor = T_W_sensor;
		R_W_sensor.transform.translation.x = 0.0;
		R_W_sensor.transform.translation.y = 0.0;
		R_W_sensor.transform.translation.z = 0.0;
		tf2::doTransform(plane_normal_msg, plane_normal_msg_W, R_W_sensor);
		Eigen::Vector3d plane_normal_W(plane_normal_msg_W.pose.position.x, plane_normal_msg_W.pose.position.y, plane_normal_msg_W.pose.position.z);

		double r = plane_normal_W.head(2).norm();
		double pitch = std::atan2(plane_normal_W(2), r);
		++counter;
		// std::cout << counter << "Pitch: " << pitch << std::endl;
		if(std::abs(pitch) >= max_normal_pitch_)
		{
			continue;
		}

		// std::cout << "Pitch ok " << std::endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		Eigen::Vector3d max(-9999, -99999, -9999), min(9999, 9999, 9999);
		for (const auto &idx : inliers->indices)
		{
			pcl::PointXYZ inlier_pt = contour_pcl->points[idx];
			inlier_cloud->insert(inlier_cloud->begin(), inlier_pt);
			Eigen::Vector4d pv(inlier_pt.x, inlier_pt.y, inlier_pt.z, getDistance(inlier_pt));
			contour_centroid += pv.head(3);
			Eigen::Vector3d pt_eigen(contour_pcl->points[idx].x, contour_pcl->points[idx].y, contour_pcl->points[idx].z);
			// Eigen::Vector3d pt_tfed = quat.toRotationMatrix() * pt_eigen + contour_centroid;
			Eigen::Vector3d pt_tfed = quat.toRotationMatrix().inverse() * pt_eigen;
			max = max.cwiseMax(pt_tfed);
			min = min.cwiseMin(pt_tfed);
			// inlier_cloud_plane->insert(inlier_cloud_plane->begin(), pcl::PointXYZ(pt_eigen.x(), pt_eigen.y(), pt_eigen.z()));
		}


		contour_centroid /= inlier_cloud->points.size();

		if(contour_centroid.norm() < min_manhole_robot_distance_) 
		{
			continue;
		}
		

		pcl::CropBox<pcl::PointXYZ> crop_box_filter(true);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
		cloud_ptr->insert(cloud_ptr->begin(), cloud.begin(), cloud.end());
		crop_box_filter.setInputCloud(cloud_ptr);
		Eigen::Vector4f min_lim(-mh_centroid_bbox_.x()/2.0, -mh_centroid_bbox_.y()/2.0, -mh_centroid_bbox_.z()/2.0, 1.0);
		Eigen::Vector4f max_lim(mh_centroid_bbox_.x()/2.0, mh_centroid_bbox_.y()/2.0, mh_centroid_bbox_.z()/2.0, 1.0);
		crop_box_filter.setMin(min_lim);
		crop_box_filter.setMax(max_lim);
		Eigen::Vector3f rpy = quat.toRotationMatrix().eulerAngles(0,1,2).cast<float>();
		crop_box_filter.setRotation(rpy);
		crop_box_filter.setTranslation(contour_centroid.cast<float>());
		pcl::PointCloud<pcl::PointXYZ> out_cloud;
		crop_box_filter.filter(out_cloud);
		*all_contours_pcl += out_cloud;

		if(out_cloud.points.size() > max_points_inside_mh_)
		{
			// ROS_WARN("MH center not free %d", out_cloud.points.size());
			continue;
		}


		/*************************/


		pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
		
		geometry_msgs::PoseStamped centroid;
		centroid.pose.position.x = contour_centroid.x();
		centroid.pose.position.y = contour_centroid.y();
		centroid.pose.position.z = contour_centroid.z();
		centroid.pose.orientation.w = quat.w();
		centroid.pose.orientation.x = quat.x();
		centroid.pose.orientation.y = quat.y();
		centroid.pose.orientation.z = quat.z();
		centroid.header.frame_id = cloud.header.frame_id;
		// centroid.header.stamp = ros::Time() cloud.header.stamp;
		pcl_conversions::fromPCL(cloud.header.stamp, centroid.header.stamp);

		Eigen::Vector3d bounding_box = (max - min);
		pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		if ((bounding_box.array() > manhole_bounding_box_size_min_.array()).all() && (bounding_box.array() < manhole_bounding_box_size_max_.array()).all())
		{
			geometry_msgs::PoseStamped centroid_W;
			tf2::doTransform(centroid, centroid_W, T_W_sensor);
			// tf_buffer_.transform(centroid, centroid_W, "world", ros::Duration(0));
			*detection_cloud += *inlier_cloud;
			++num_manholes;
			Eigen::Vector3d centroid_pos_W(centroid_W.pose.position.x, centroid_W.pose.position.y, centroid_W.pose.position.z);
			Eigen::Quaterniond quat_W;
			quat_W.x() = centroid_W.pose.orientation.x;
			quat_W.y() = centroid_W.pose.orientation.y;
			quat_W.z() = centroid_W.pose.orientation.z;
			quat_W.w() = centroid_W.pose.orientation.w;
			// std::cout << "Detection yaw: " << quat_W.toRotationMatrix().eulerAngles(0, 1, 2)(2) << std::endl;
			detections.push_back(std::make_pair(centroid_pos_W, quat_W.toRotationMatrix().eulerAngles(0, 1, 2)(2)));

			// centroids.poses.push_back(centroid.pose);
			centroids.poses.push_back(centroid_W.pose);
			pub_contour_centroid_.publish(centroids);

			if((manhole_pos - Eigen::Vector3d(centroid_W.pose.position.x, centroid_W.pose.position.y, centroid_W.pose.position.z)).norm() <= 0.4)
			{
				nav_msgs::Odometry detection_odom;
				detection_odom = current_odom_;
				pub_detection_odom_.publish(detection_odom);
			}
		}

		for (auto pt : detection_cloud->points)
		// for (auto pt : inlier_cloud_plane->points)
		// for (auto pt : contour_pcl->points)
		{
			pcl::PointXYZI pti;
			pti.x = pt.x;
			pti.y = pt.y;
			pti.z = pt.z;
			pti.intensity = counter * 100;
			all_contours_pcl_intensity->points.push_back(pti);
		}
	}

	for (auto det : detections)
	{
		// std::cout << det.first.transpose() << " " << det.second << std::endl;
		double closest_manhole_dist = std::numeric_limits<double>::max();
		int closest_manhole_id = -1;
		for (int i = 0; i < manhole_detections_.size(); ++i)
		{
			double dist = (det.first - manhole_detections_[i].mean_pos).norm();
			if (dist < cluster_radius_)
			{
				if (dist < closest_manhole_dist) // New closer manhole found
				{
					closest_manhole_dist = dist;
					closest_manhole_id = i;
				}
			}
		}
		if (closest_manhole_id >= 0) // Part of an existing manhole
		{
			// if (!manhole_detections_[closest_manhole_id].seen_sufficiently || 
			// 	(manhole_detections_[closest_manhole_id].seen_sufficiently && 
			// 		manhole_detections_[closest_manhole_id].detections.size() < min_detection_to_be_permenant_*5))
			manhole_detections_[closest_manhole_id].detection_buffer.insert(manhole_detections_[closest_manhole_id].detection_buffer.begin(), 1);
			manhole_detections_[closest_manhole_id].detections.push_back(Eigen::Vector4d(det.first(0), det.first(1), det.first(2), det.second));
			if(manhole_detections_[closest_manhole_id].detection_buffer.size() > min_detection_to_be_permenant_*5)
			{
				manhole_detections_[closest_manhole_id].detection_buffer.erase(manhole_detections_[closest_manhole_id].detection_buffer.begin());
			}
			if(manhole_detections_[closest_manhole_id].detections.size() > min_detection_to_be_permenant_*5)
			{
				manhole_detections_[closest_manhole_id].detections.erase(manhole_detections_[closest_manhole_id].detections.begin());
			}
			for (int i = 0; i < manhole_detections_.size(); ++i)
			{
				if (i != closest_manhole_id)
				{
					manhole_detections_[i].detection_buffer.insert(manhole_detections_[i].detection_buffer.begin(), 0);
				}
				if (manhole_detections_[i].detection_buffer.size() >= buffer_size_)
				{
					manhole_detections_[i].detection_buffer.pop_back();
				}
			}
		}
		else // New manhole to be created
		{
			for (int i = 0; i < manhole_detections_.size(); ++i)
			{
				manhole_detections_[i].detection_buffer.insert(manhole_detections_[i].detection_buffer.begin(), 0);
				if (manhole_detections_[i].detection_buffer.size() >= buffer_size_)
				{
					manhole_detections_[i].detection_buffer.pop_back();
				}
			}

			ManholeDetection new_manhole;
			new_manhole.id = manhole_count_;
			manhole_count_ += 59;
			new_manhole.detection_buffer.insert(new_manhole.detection_buffer.begin(), 1);
			new_manhole.mean_pos = det.first;
			new_manhole.mean_yaw = det.second;
			new_manhole.detections.push_back(Eigen::Vector4d(det.first(0), det.first(1), det.first(2), det.second));
			manhole_detections_.push_back(new_manhole);
		}
	}

	std::vector<ManholeDetection> detections_copy;
	std::vector<ManholeDetection> new_stable_detections;
	for (auto det : manhole_detections_)
	{
		if (!det.seen_sufficiently)
		{
			bool new_stable_detection = false;
			if (det.detections.size() > min_detection_to_be_permenant_)
			{
				det.seen_sufficiently = true;
				new_stable_detection = true;
			}
			std::vector<int>::iterator it = std::find(det.detection_buffer.begin(), det.detection_buffer.end(), 1);
			if (it != det.detection_buffer.end() || det.seen_sufficiently) // At least one detection of this hole in last `buffer_size_` meansurements
			{
				Eigen::Vector3d new_mean_pos = Eigen::Vector3d::Zero();
				double new_mean_yaw = -5*M_PI;
				// std::cout << "Detection: " << det.id << std::endl;
				for (int di=0; di<det.detections.size(); ++di)
				{
					auto pt = det.detections[di];
					new_mean_pos += pt.head(3);
					double c_yaw = pt(3);
					correctDetectionYaw(c_yaw);
					// std::cout << c_yaw << ", ";
					// new_mean_yaw += c_yaw;
					if(di==0)
					{
						new_mean_yaw = c_yaw;
						// truncateYaw(new_mean_yaw);
					}
					else
					{
						correctDetectionYaw(c_yaw, new_mean_yaw);
						new_mean_yaw *= di;
						new_mean_yaw += c_yaw;
						new_mean_yaw /= (di+1);
					}
					// std::cout << "(" << c_yaw << ", " << new_mean_yaw << ") | ";
				}
				// std::cout << std::endl;
				new_mean_pos /= det.detections.size();
				// new_mean_yaw /= det.detections.size();
				det.mean_pos = new_mean_pos;
				det.mean_yaw = new_mean_yaw;
				detections_copy.push_back(det);
				if (new_stable_detection)
				{
					// std::cout << "New stable detection: ID: " << det.id << std::endl;
					new_stable_detections.push_back(det);
				}
			}
		}
		else
		{
			Eigen::Vector3d new_mean_pos = Eigen::Vector3d::Zero();
			double new_mean_yaw = -5*M_PI;
			// std::cout << "Detection: " << det.id << std::endl;
			for (int di=0; di<det.detections.size(); ++di)
			{
				auto pt = det.detections[di];
				new_mean_pos += pt.head(3);
				double c_yaw = pt(3);
				correctDetectionYaw(c_yaw);
				// std::cout << c_yaw << ", ";
				// new_mean_yaw += c_yaw;
				if(di==0)
				{
					new_mean_yaw = c_yaw;
					// truncateYaw(new_mean_yaw);
				}
				else
				{
					correctDetectionYaw(c_yaw, new_mean_yaw);
					new_mean_yaw *= di;
					new_mean_yaw += c_yaw;
					new_mean_yaw /= (di+1);
				}
				// std::cout << "(" << c_yaw << ", " << new_mean_yaw << ") | ";
			}
			// std::cout << std::endl;
			new_mean_pos /= det.detections.size();
			// new_mean_yaw /= det.detections.size();
			det.mean_pos = new_mean_pos;
			det.mean_yaw = new_mean_yaw;
			detections_copy.push_back(det);
		}
	}

	manhole_detections_ = detections_copy;

	planner_msgs::MultipleManholeDetections detection_message;
	// std::cout << "Publishing detections:\n";
	for (int i = 0; i < new_stable_detections.size(); ++i)
	{
		planner_msgs::ManholeDetection det;
		det.id = new_stable_detections[i].id;

		tf::Quaternion quat;
		quat.setEuler(0.0, 0.0, new_stable_detections[i].mean_yaw);
		tf::Vector3 origin(new_stable_detections[i].mean_pos[0], new_stable_detections[i].mean_pos[1], new_stable_detections[i].mean_pos[2]);
		tf::Pose poseTF(quat, origin);
		geometry_msgs::Pose pose;
		tf::poseTFToMsg(poseTF, pose);
		det.pose = pose;
		detection_message.multiple_detections.push_back(det);
	}

	geometry_msgs::PoseArray stable_detections;
	stable_detections.header.frame_id = "world";
	planner_msgs::MultipleManholeDetections stable_detections_msg;
	// std::cout << "Vis: " << std::endl;
	for (auto det : manhole_detections_)
	{
		if (det.seen_sufficiently)
		{
			planner_msgs::ManholeDetection det_msg;
			det_msg.id = det.id;
			geometry_msgs::Pose pose;
			pose.position.x = det.mean_pos.x();
			pose.position.y = det.mean_pos.y();
			pose.position.z = det.mean_pos.z();
			tf2::Quaternion quat;
			quat.setEuler(0.0, 0.0, det.mean_yaw);
			pose.orientation.x = quat.x();
			pose.orientation.y = quat.y();
			pose.orientation.z = quat.z();
			pose.orientation.w = quat.w();
			stable_detections.poses.push_back(pose);
			det_msg.pose = pose;

			stable_detections_msg.multiple_detections.push_back(det_msg);
		}
	}

	// std::cout << "------------" << std::endl;
	pub_detections_vis_.publish(stable_detections);
	pub_detections_.publish(stable_detections_msg);
	
		if(!first_detection_ && !stable_detections.poses.empty()) {
		first_detection_ = true;
		first_mh_location_ << stable_detections.poses[0].position.x, stable_detections.poses[0].position.y, stable_detections.poses[0].position.z;
		ROS_ERROR("First MH: %f, %f, %f: ", stable_detections.poses[0].position.x, stable_detections.poses[0].position.y, stable_detections.poses[0].position.z);
	}

	if(first_detection_)
	{
		bool det_success = false;
		for(auto c : centroids.poses) 
		{
			Eigen::Vector3d det(c.position.x, c.position.y, c.position.z);
			if((first_mh_location_ - det).norm() < 1.0)
			{
				det_success = true;
				break;
			}
		}
		geometry_msgs::PoseArray dummy_pose;
		dummy_pose.header = in_cloud.header;
		if(det_success)
		{
			geometry_msgs::Pose p;
			// p.position.x = 1.0;
			// dummy_pose.poses.push_back(p);
			geometry_msgs::PoseStamped ps, psw;
			ps.pose.orientation.w = 1.0;
			tf2::doTransform(ps, psw, T_W_sensor);
			detection_attempt_poses_.poses.push_back(psw.pose);
		}
		else
		{
			// geometry_msgs::Pose p;
			// p.position.x = -1.0;
			// dummy_pose.poses.push_back(p);
		}
		detection_attempt_poses_.header.frame_id = "world";
		pub_processed_cloud_.publish(detection_attempt_poses_);
	}

	sensor_msgs::PointCloud2 cloud_msg;
	all_contours_pcl->header.frame_id = cloud.header.frame_id;
	pcl::toROSMsg(*all_contours_pcl_intensity, cloud_msg);
	cloud_msg.header.frame_id = cloud.header.frame_id;
	
	pub_contour_pcl_.publish(cloud_msg);
}

void ManholeDetector::reEvaluate(int id) {
	ManholeDetection test_det;
	test_det.id = id;
	auto it = std::find_if(manhole_detections_.begin(), manhole_detections_.end(), 
						[&test_det](const ManholeDetection& x) { return x.id == test_det.id;});
	if(it != manhole_detections_.end()) {
		it->seen_sufficiently = false;
		it->detections.clear();
	}
}

bool ManholeDetector::loadParams()
{
	std::string ns = ros::this_node::getName();
	ROS_INFO("Loading: %s", ns.c_str());

	std::string param_name;

	param_name = ns + "/cluster_radius";
	nh_.param<double>(param_name, cluster_radius_, 2.0);
	std::cout << "cluster_radius: " << cluster_radius_ << std::endl;

	param_name = ns + "/canny_low_thres";
	nh_.param<int>(param_name, canny_low_thres_, 20);
	std::cout << "canny_low_thres: " << canny_low_thres_ << std::endl;

	param_name = ns + "/slack";
	nh_.param<int>(param_name, slack_, 2);
	std::cout << "slack: " << slack_ << std::endl;

	param_name = ns + "/col_offset";
	nh_.param<int>(param_name, col_offset_, 20);
	std::cout << "col_offset: " << col_offset_ << std::endl;

	param_name = ns + "/row_offset";
	nh_.param<int>(param_name, row_offset_, 10);
	std::cout << "row_offset: " << row_offset_ << std::endl;

	param_name = ns + "/buffer_size";
	nh_.param<int>(param_name, buffer_size_, 10);
	std::cout << "buffer_size: " << buffer_size_ << std::endl;

	param_name = ns + "/min_detection_to_be_permenant";
	nh_.param<int>(param_name, min_detection_to_be_permenant_, 4);
	std::cout << "min_detection_to_be_permenant: " << min_detection_to_be_permenant_ << std::endl;

	param_name = ns + "/min_manhole_robot_distance";
	nh_.param<double>(param_name, min_manhole_robot_distance_, 0.3);
	std::cout << "min_manhole_robot_distance: " << min_manhole_robot_distance_ << std::endl;

	param_name = ns + "/max_normal_pitch";
	nh_.param<double>(param_name, max_normal_pitch_, 0.78);
	std::cout << "max_normal_pitch: " << max_normal_pitch_ << std::endl;

	param_name = ns + "/crop_out_rows";
	nh_.param<bool>(param_name, crop_out_rows_, false);
	std::cout << "crop_out_rows: " << crop_out_rows_ << std::endl;

	param_name = ns + "/max_points_inside_mh";
	nh_.param<int>(param_name, max_points_inside_mh_, 2);
	std::cout << "max_points_inside_mh: " << max_points_inside_mh_ << std::endl;

	param_name = ns + "/use_px_offsets";
	nh_.param<bool>(param_name, use_px_offsets_, false);
	std::cout << "use_px_offsets: " << use_px_offsets_ << std::endl;

	param_name = ns + "/num_laser_beams";
	nh_.param<int>(param_name, num_laser_beams_, 64);
	std::cout << "num_laser_beams: " << num_laser_beams_ << std::endl;

	std::vector<double> param_val;
	param_name = ns + "/manhole_bounding_box_size_min";
	if ((!ros::param::get(param_name, param_val)) || (param_val.size() != 3))
	{
		manhole_bounding_box_size_min_ << 0.0, 0.5, 0.7;
	}
	else
	{
		manhole_bounding_box_size_min_ << param_val[0], param_val[1], param_val[2];
	}

	param_val.clear();
	param_name = ns + "/mh_centroid_bbox";
	if ((!ros::param::get(param_name, param_val)) || (param_val.size() != 3))
	{
		mh_centroid_bbox_ << 1.0, 0.3, 0.3;
	}
	else
	{
		mh_centroid_bbox_ << param_val[0], param_val[1], param_val[2];
	}

	param_val.clear();
	param_name = ns + "/manhole_bounding_box_size_max";
	if ((!ros::param::get(param_name, param_val)) || (param_val.size() != 3))
	{
		manhole_bounding_box_size_max_ << 0.6, 0.9, 1.1;
	}
	else
	{
		manhole_bounding_box_size_max_ << param_val[0], param_val[1], param_val[2];
	}

	std::vector<int> param_val_int;
	param_name = ns + "/px_offsets";
	if (use_px_offsets_ && (ros::param::get(param_name, param_val_int)) && (param_val_int.size() == 64))
	{
		px_offsets_ = param_val_int;
	}
	else
	{
		px_offsets_ = std::vector<int>(num_laser_beams_,0);
	}

	// std::vector<int> param_val_int;
	param_name = ns + "/indices_to_crop";
	if ((!ros::param::get(param_name, param_val_int)) || (param_val_int.size() != 2))
	{
		indices_to_crop_ << 0, 0;
	}
	else
	{
		indices_to_crop_ << param_val_int[0], param_val_int[1];
	}
	return true;
}

bool ManholeDetector::getStableManholes(std::vector<ManholeDetection> &detections)
{
	detections.clear();

	if(manhole_detections_.empty())
	{
		return false;
	}

	for(auto det : manhole_detections_)
	{
		if(det.seen_sufficiently)
		{
			detections.push_back(det);
		}
	}

	return true;
}

void ManholeDetector::odomCallback(const nav_msgs::Odometry &odom)
{
	current_odom_ = odom;
}