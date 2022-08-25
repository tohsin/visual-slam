#include <iostream>
#include <opencv2/opencv.hpp>
std::string img1 = "/Users/emma/dev/visual-slam/orb_extraction_cv/data/1.png";
std::string img2 = "/Users/emma/dev/visual-slam/orb_extraction_cv/data/2.png";

using namespace  std;
using namespace  cv;

void find_feature_matches(
        const Mat &img_1,
        const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

void pose_estimation_2d2d(
        const std::vector<KeyPoint> &keypoints_1,
        const std::vector<KeyPoint> &keypoints_2,
        const std::vector<DMatch> &matches,
        Mat &R, Mat &t);

void triangulation(
        const vector<KeyPoint> &keypoint_1,
        const vector<KeyPoint> &keypoint_2,
        const std::vector<DMatch> &matches,
        const Mat &R,
        const Mat &t,
        vector<Point3d> &points
);
// Pixel coordinates to camera normalized coordinates
cv::Point2d pixel2cam(const Point2d &p, const Mat &K);

inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}
int main() {
    cv::Mat img_1 = imread(img1, cv::IMREAD_COLOR);
    cv::Mat img_2 = imread(img2, cv::IMREAD_COLOR);
    std::vector<KeyPoint> keypoints_1 , keypoints_2;
    vector<DMatch> matches;

    find_feature_matches( img_1 = img_1, img_2 = img_2 ,
                          keypoints_1 = keypoints_1, keypoints_2 = keypoints_2 , matches = matches);

    cv::Mat R, t;
    pose_estimation_2d2d(   keypoints_1, keypoints_2, matches, R, t);

    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);


    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1,
                                                0, 521.0, 249.7,
                                                0, 0, 1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();


    //-- Verify the reprojection relationship between the triangulation point and the feature point
    for (int i = 0; i < matches.size(); i++) {
        //first picture
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt,
                   2, get_color(depth1), 2);

        // second figure
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2,
                   get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();
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
        const std::vector<KeyPoint> &keypoints_1,
        const std::vector<KeyPoint> &keypoints_2,
        const std::vector<DMatch> &matches,
        Mat &R,
        Mat &t){
    //cv::Mat_ can be more convenient if you use a lot of element access operations
    // and if you know matrix type at compile time.
    // Note that cv::Mat::at<_Tp>(int y, int x) and
    // cv::Mat_<_Tp>::operator ()(int y, int x)
    // do absolutely the same thing and run at the same speed, but the latter is certainly shorter:
    // Camera Intrinsics,TUM Freiburg2
    cv::Mat K = ( cv::Mat_<double>(3, 3) <<  520.9, 0,     325.1,
                                                        0,     521.0, 249.7,
                                                        0,     0,     1     );


    //Convert the matching point to the form of vector<Point2f>
    vector<cv::Point2f> points_1, points_2;

    for (int i = 0 ; i < (int) matches.size() ; i ++){

        points_1.push_back( keypoints_1[matches[i].queryIdx].pt );
        points_2.push_back( keypoints_2[matches[i].trainIdx].pt );
    }



    //−− Calculate essential matrix
    // camera principal point, calibrated in TUM dataset
    cv::Point2d principal_point(325.1, 249.7);

    // camera focal length, calibrated in TUM dataset
    double focal_length = 521;

    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points_1, points_2, focal_length, principal_point);

    // essential matrics is complied from u and v to get  a value varied from u , v and points1 and points 2



    //−− Recover rotation and translation from the essential matrix.
    cv::recoverPose(essential_matrix, points_1, points_2, R, t, focal_length,
                    principal_point);

}

void triangulation(
        const vector<KeyPoint> &keypoint_1,
        const vector<KeyPoint> &keypoint_2,
        const std::vector<DMatch> &matches,
        const Mat &R,
        const Mat &t,
        vector<Point3d> &points
){
    // declare T1 and T2
    cv::Mat T1 = (cv::Mat_<float>(3, 4)<<  1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0);
    // transformation matrix generated from R and t ([R,t], [0,1]
    cv::Mat T2 = (Mat_<float>(3, 4) <<
                                R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    cv::Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> pts_1, pts_2;
    for (DMatch m:matches) {
        // converting pixel coorinte to cam cordiates
        pts_1.push_back(  pixel2cam(keypoint_1[m.queryIdx].pt, K)  );
        pts_2.push_back(   pixel2cam(keypoint_2[m.trainIdx].pt, K)     );
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for (int i = 0; i < pts_4d.cols; i++) {
        // Convert to non−homogeneous coordinates
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // normalised

        cv::Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
        );
        points.push_back(p);
    }


}
cv::Point2d pixel2cam(const Point2d &p, const Mat &K) {

    return Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

