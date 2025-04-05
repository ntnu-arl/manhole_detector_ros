#include "manhole_detector/manhole_detector.hpp"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "manhole_detector");
	ros::NodeHandle nh;
	ros::NodeHandle nh_private;
    ManholeDetector detector(nh, nh_private);

	ros::spin();
	
    return 0;
}
