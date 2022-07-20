#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat &img_1,
        const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat &R,
        Mat &t);

cv::Point2d pixel2cam(const Point2d &p, const Mat &K);

std::string img1 = "/Users/emma/dev/visual-slam/orb_extraction_cv/data/1.png";
std::string img2 = "/Users/emma/dev/visual-slam/orb_extraction_cv/data/2.png";
int main() {
    cv::Mat img_1 = imread(img1, cv::IMREAD_COLOR);
    cv::Mat img_2 = imread(img2, cv::IMREAD_COLOR);

    assert(img_1.data && img_2.data && "Can not load images!");
    std::vector<KeyPoint> keypoints_1 , keypoints_2;
    vector<DMatch> matches;

    find_feature_matches( img_1 = img_1, img_2 = img_2 ,
                          keypoints_1 = keypoints_1, keypoints_2 = keypoints_2 , matches = matches);

    cout << "number of feature matches in both images " << matches.size() << " set of feature points" << endl;

    cv::Mat R, t;
    // get the odometry motion between the two frames
    pose_estimation_2d2d(keypoints_1 , keypoints_2 , matches , R , t);
    // check if E = t^R * scale
    cv::Mat t_x = ( cv::Mat_<double>(3,3) << 0, -t.at<double>(2, 0) , t.at<double>(1,0),
                                        t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                                        -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R=" << endl << t_x * R << endl;
    //−− Check epipolar constraints
    cv::Mat K = (Mat_<double>(3, 3) <<   520.9, 0,     325.1,
                                                    0,     521.0, 249.7,
                                                    0,        0,      1);
    for (DMatch m: matches) {
        cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = ( Mat_<double>(3, 1) <<  pt1.x,  pt1.y,  1);

        cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() *  t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;




    return 0;
}


void find_feature_matches(
        const Mat &img_1,
        const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches){
    //matches is variable to save the matches from key poitns
    cv::Mat descriptor_1,descriptor_2;


    cv::Ptr<FeatureDetector> detector = ORB::create();
    cv::Ptr<DescriptorExtractor> descriptor = ORB::create();

    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);


    descriptor->compute(img_1, keypoints_1, descriptor_1);
    descriptor->compute(img_2, keypoints_2, descriptor_2);

    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptor_1, descriptor_2, match);


    double min_dist = 10000, max_dist = 0;


    for (int i = 0; i < descriptor_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);


    for (int i = 0; i < descriptor_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat &R,
        Mat &t){
    //cv::Mat_ can be more convenient if you use a lot of element access operations
    // and if you know matrix type at compile time.
    // Note that cv::Mat::at<_Tp>(int y, int x) and
    // cv::Mat_<_Tp>::operator ()(int y, int x)
    // do absolutely the same thing and run at the same speed, but the latter is certainly shorter:
    // Camera Intrinsics,TUM Freiburg2
    cv::Mat K = ( cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );


    //Convert the matching point to the form of vector<Point2f>
    vector<cv::Point2f> points_1, points_2;

    for (int i = 0 ; i < (int) matches.size() ; i ++){

        points_1.push_back( keypoints_1[matches[i].queryIdx].pt );
        points_2.push_back( keypoints_2[matches[i].trainIdx].pt );
    }

    // compute the fundamental matrix
    //  F=K−T E K−1
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points_1, points_2, cv::FM_8POINT);

    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //−− Calculate essential matrix
    // camera principal point, calibrated in TUM dataset
    cv::Point2d principal_point(325.1, 249.7);

    // camera focal length, calibrated in TUM dataset
    double focal_length = 521;

    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points_1, points_2, focal_length, principal_point);

    // essential matrics is complied from u and v to get  a value varied from u , v and points1 and points 2
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    //−− Calculate homography matrix
    //−− But the scene is not planar, and calculating the homography matrix here is little significance
    cv::Mat homography_matrix;

    homography_matrix = cv::findHomography(points_1, points_2, cv::RANSAC, 3);
    std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;

    //−− Recover rotation and translation from the essential matrix.
    cv::recoverPose(essential_matrix, points_1, points_2, R, t, focal_length,
                                                                                principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

}


cv::Point2d pixel2cam(const Point2d &p, const Mat &K) {

    return Point2d(
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}
