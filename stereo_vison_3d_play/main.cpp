#include <iostream>

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <unistd.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pangolin/var/var.h>
#include <pangolin/var/varextra.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>

using  namespace  Eigen;
using  namespace std;

std::string left_file = "/Users/emma/dev/visual-slam/stereo_vison_3d_play/data/left.png";
std::string right_file = "/Users/emma/dev/visual-slam/stereo_vison_3d_play/data/right.png";

void showPointCloud(
        const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main() {
    // declare variables used in stereo vison
    double base_line = 0.573;
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);

    // disparity variables to construct disparity or like depth maps
    //SGBM (Semi-global Batch Matching) [26] algorithm implemented by OpenCV to
    // calculate the disparity of the left and right images
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            0,96,9,8*9*9,32*9*9,
            1,63,10,100,32);
    // honestly still not sure what that does tbh


    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);


    std::vector<Vector4d, Eigen::aligned_allocator<Vector4d>> point_cloud;

    for (int v = 0; v < left.rows; v++){
        for (int u = 0; u < left.cols; u++) {
            // setting limits for disparity  of pixels
            std::cout << "Disparity and left data "<<std::endl;
            std::cout << disparity.at<float>(v, u) << left.at<uchar>(v, u) << std::endl;
            if (disparity.at<float>(v, u) <= 10.0 || disparity.at<float>(v, u) >= 96.0)
                continue;


            // the first three dimensions are xyz, the 4−th is the color
            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0);

            // calculate depth from disparity
            double x = (u - cx)/ fx;
            double y = (v - cy)/fy;
            // z = fb / d  , d = ul - ur or depsarity
            double depth = fx * base_line / (disparity.at<float>(v, u));
            point [0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            point_cloud.push_back(point);

        }
    }
    // display disparity
//    cv::imshow("disparity", disparity / 96.0);
//    cv::waitKey(0);

    showPointCloud(point_cloud);

    return 0;
}
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}


