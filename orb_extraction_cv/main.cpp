#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;
using namespace cv;

std::string img1 = "/Users/emma/dev/visual-slam/orb_extraction_cv/data/1.png";
std::string img2 = "/Users/emma/dev/visual-slam/orb_extraction_cv/data/2.png";

int main() {


    cv::Mat img_1 = imread(img1, cv::IMREAD_COLOR);
    cv::Mat img_2 = imread(img2, cv::IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // Keypoint is  data structure to store points used in orb for example
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();


    // detect features abd send to key points
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);


    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;


    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg1);



//    cv::waitKey(0);
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("−− Max dist : %f \n", max_dist);
    printf("−− Min dist : %f \n", min_dist);

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
            }
    }

    cv::Mat img_match;
    cv::Mat img_goodmatch;

    cout<< matches.size()<<endl;
    cout<< good_matches.size()<<endl;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2,
                    matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2,
                   keypoints_2, good_matches, img_goodmatch);
    cv::imshow("all matches", img_match);
    cv::imshow("good matches", img_goodmatch);
    cv::waitKey(0);



    return 0;
}
