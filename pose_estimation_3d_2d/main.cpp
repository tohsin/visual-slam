#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <chrono>


using  namespace  std;

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

Eigen::Matrix3d compute_rotation_matrix(Eigen::Vector3d const omega );

Eigen::Vector3d  compute_translation_matric(Eigen::Vector3d rho,Eigen::Vector3d  omega);
void find_feature_matches(
        const cv::Mat &img_1, const cv::Mat &img_2,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::KeyPoint> &keypoints_2,
        std::vector<cv::DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const cv::Mat &K,
        Sophus::SE3d &pose
        //  Eigen::Matrix<double, 4, 4>  &pose
);




std::string img1 = "/Users/emma/dev/visual-slam/pose_estimation_3d_2d/1.png";
std::string img2 = "/Users/emma/dev/visual-slam/pose_estimation_3d_2d/2.png";
std::string img3 = "/Users/emma/dev/visual-slam/pose_estimation_3d_2d/1_depth.png";
int main() {


    cv::Mat img_1 = imread(img1, cv::IMREAD_COLOR);
    cv::Mat img_2 = imread(img2, cv::IMREAD_COLOR);


    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

    //The depth map is a 16-bit unsigned number, a single-channel image
    cv::Mat d1 = imread(img3, cv::IMREAD_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    for (cv::DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
        continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;



    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::Mat r, t;
    //Call OpenCV's PnP solution, choose EPNP, DLS and other methods
    solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Mat R;

    //r is in the form of a rotation vector, converted to a matrix using the Rodrigues formula
    cv::Rodrigues(r, R);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;


    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;

    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by gauss newton" << endl;
    Sophus::SE3d pose_gn;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    // Eigen::Matrix<double, 4, 4>  pose_gn;
    // pose_gn << 1, 0,0,0,
    //             0,1,0,0,
    //             0,0,1,0,
    //             0,0,0,1;
    // t1 = chrono::steady_clock::now();
    // bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

    return 0;
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {

    return cv::Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}


void find_feature_matches(
        const cv::Mat &img_1,
        const cv::Mat &img_2,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::KeyPoint> &keypoints_2,
        std::vector<cv::DMatch> &matches){
    //matches is variable to save the matches from key poitns
    cv::Mat descriptor_1,descriptor_2;


    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);


    descriptor->compute(img_1, keypoints_1, descriptor_1);
    descriptor->compute(img_2, keypoints_2, descriptor_2);

    vector<cv::DMatch> match;
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




void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const  cv::Mat &K,
  Sophus::SE3d &pose) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;
  double cost = 0, lastCost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; iter++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // compute cost
    for (int i = 0; i < points_3d.size(); i++) {
      Eigen::Vector3d pc = pose * points_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

      Eigen::Vector2d e = points_2d[i] - proj;

      cost += e.squaredNorm();
      Eigen::Matrix<double, 2, 6> J;
      J << -fx * inv_z,
        0,
        fx * pc[0] * inv_z2,
        fx * pc[0] * pc[1] * inv_z2,
        -fx - fx * pc[0] * pc[0] * inv_z2,
        fx * pc[1] * inv_z,
        0,
        -fy * inv_z,
        fy * pc[1] * inv_z2,
        fy + fy * pc[1] * pc[1] * inv_z2,
        -fy * pc[0] * pc[1] * inv_z2,
        -fy * pc[0] * inv_z;

      H += J.transpose() * J;
      b += -J.transpose() * e;
    }

    Vector6d dx;
    dx = H.ldlt().solve(b);

    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // update your estimation
    Eigen::Vector3d omega = dx.template tail<3>();
    Eigen::Vector3d rho = dx.template head<3>();
    Eigen::Matrix3d rotation_matrix = compute_rotation_matrix(omega);
    Eigen::Vector3d translation_part = compute_translation_matric(rho, omega);
    std::cout<<"#################### \n"<< std::endl;
    std::cout<<"rotation matrix \n"<<rotation_matrix << std::endl;
    std::cout<<"translation matrix \n"<<translation_part << std::endl;

    Eigen::Matrix<double, 4, 4> transformation_matrix;

    transformation_matrix<< rotation_matrix(0,0), rotation_matrix(0,1), rotation_matrix(0,2), translation_part(0),
                            rotation_matrix(1,0), rotation_matrix(1,1), rotation_matrix(1,2), translation_part(1),
                            rotation_matrix(2,0), rotation_matrix(2,1),rotation_matrix(2,2), translation_part(2),
                            0,0,0,1;
    std::cout<<"Transformation matrix \n"<<transformation_matrix << std::endl;
//    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
    if (dx.norm() < 1e-6) {
      // converge
      break;
    }
  }

  cout << "pose by g-n: \n" << pose.matrix() << endl;
}



// void bundleAdjustmentGaussNewton(
//         const VecVector3d &points_3d,
//         const VecVector2d &points_2d,
//         const cv::Mat &K,
//         Eigen::Matrix<double, 4, 4> & pose
//         // Sophus::SE3d &pose
// ) {
//     typedef Eigen::Matrix<double, 6, 1> Vector6d;
//     //number of iterat
//     const int iterations = 10;
//     double cost = 0, lastCost = 0;
//     double fx = K.at<double>(0, 0);
//     double fy = K.at<double>(1, 1);
//     double cx = K.at<double>(0, 2);
//     double cy = K.at<double>(1, 2);

//     for (int iter = 0; iter < iterations; iter++) {
//         // declare the two variables for Least square optimisation
//         Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
//         Vector6d b = Vector6d::Zero();
//         // intilaise chi error
//         chi_error = 0;
//         // error is between projected value and  projected pixed location
//         // we wish to minimize the whole graph or all possible projected points
//         for (int i = 0; i < points_3d.size(); i++) {
//             // value for P projected position , 3d position  by pose of camera
//             // first convet point 3d to 4d by adding 1
//             Eigen::Matrix<double, 4, 1>  point3d_4;
//             point3d_4 << points_3d[i][0], points_3d[i][1], points_3d[i][2],1;

//             Eigen::Matrix<double, 4, 1> p_c_pre =  pose * point3d_4;
//             Eigen::Vector3d projected_cordinates;
//             projected_cordinates[0] = p_c_pre[0];
//             projected_cordinates[1] = p_c_pre[1];
//             projected_cordinates[2] =p_c_pre[2];

//             double X = projected_cordinates[0];
//             double Y = projected_cordinates[1];
//             double Z = projected_cordinates[2];
//             //obtain variables for Jacobian matrix
//             double inverse_z = 1 / projected_cordinates[2]; // 1 / z
//             double inverse_z_squared = inverse_z * inverse_z; // 1 / Z^2
//             // camera projection u= [u,v] using camera intristics
//             Eigen::Vector2d camera_proj(fx * projected_cordinates[0] / projected_cordinates[2] + cx,
//                                         fy * projected_cordinates[1] / projected_cordinates[2] + cy);
//             // error is diffrence with camera from projected point and actual projected point
//             Eigen::Vector2d error = points_2d[i] - camera_proj;
//             // now we minimize the error
//             //error is mse loss error^squared
//             chi_error += error.squaredNorm();

//             // to minimize we obtain jacaobian
//             // equation 6.46 page 159

//             Eigen::Matrix<double, 2, 6> jacobian;
//             // ultipy through by -
//             jacobian << -fx * inverse_z,
//                     0,
//                     -fx * X * inverse_z_squared,
//                     fx * X * Y * inverse_z,
//                     -fx - fx * X * X * inverse_z_squared,
//                     fx * Y * inverse_z,
//                     0,
//                     -fy * inverse_z,
//                     fy * Y * inverse_z_squared,
//                     fy + fy * Y * Y * inverse_z_squared,
//                     -fy * X * Y * inverse_z_squared,
//                     -fy * X * inverse_z;
//             // using the hjacobian to obtain H and b H= J * J.T and b = -J * f(x)
//             H += jacobian.transpose() * jacobian;
//             b += -jacobian.transpose() * error;
//         }

// //        // compute delta x to add to x and optimise
//         Vector6d dx;
//         dx = H.ldlt().solve(b);
//          if (isnan(dx[0])) {
//             cout << "result is nan!" << endl;
//             break;
//         }

//          if (iter > 0 && chi_error >= lastCost) {
//             // cost increase, update is not good
//             cout << "cost: " << chi_error << ", last cost: " << lastCost << endl;
//             break;
//         }
// //

//         Eigen::Vector3d omega = dx.template tail<3>();
//         Eigen::Vector3d rho = dx.template head<3>();
//         Eigen::Matrix3d rotation_matrix = compute_rotation_matrix(omega);
//         Eigen::Vector3d translation_part = compute_translation_matric(rho, omega);
//         std::cout<<"#################### \n"<< std::endl;
//         std::cout<<"rotation matrix \n"<<rotation_matrix << std::endl;
//         std::cout<<"translation matrix \n"<<translation_part << std::endl;

//         Eigen::Matrix<double, 4, 4> transformation_matrix;

//         transformation_matrix<< rotation_matrix(0,0), rotation_matrix(0,1), rotation_matrix(0,2), translation_part(0),
//                             rotation_matrix(1,0), rotation_matrix(1,1), rotation_matrix(1,2), translation_part(1),
//                             rotation_matrix(2,0), rotation_matrix(2,1),rotation_matrix(2,2), translation_part(2),
//                             0,0,0,1;
//         std::cout<<"Transformation matrix \n"<<transformation_matrix << std::endl;

//         pose = transformation_matrix * pose;
      
// //        // book keeping
//         lastCost = chi_error;
       
//         cout << "iteration " << iter << " cost=" << std::setprecision(12) << chi_error << endl;
//         if (dx.norm() < 1e-6) {
//             // converge
//             break;
//         }


  
// //

//         // output is optimised pse of points

//     }
//     cout << "pose by g-n: \n" << pose << endl;
// }
Eigen::Vector3d  compute_translation_matric(Eigen::Vector3d rho,Eigen::Vector3d  omega){
    /// Translation = J * rho
    double theta_sq = omega.squaredNorm();
    double theta = std::sqrt(theta_sq);
    double theta_tripled = theta_sq * theta; 
    Eigen::Matrix3d Identity_matrix = Eigen::Matrix3d::Identity();
    double sine_ = std::sin(theta);
    double cos_ = std::cos(theta);

    Eigen::Matrix3d skew_theta;
    double omega_1 = omega[0];
    double omega_2 = omega[1];
    double omega_3 = omega[2];
    skew_theta << 0 ,     -omega_3 , omega_2,
                omega_3,  0        , -omega_1,
                -omega_2, omega_1  ,  0;

    Eigen::Matrix3d skew_theta_squared = skew_theta * skew_theta;

    double factor_1 = (1 - cos_) / theta_sq;
    double factor_2 = (theta - sine_) /theta_tripled;

    Eigen::Matrix3d first_part = factor_1 * skew_theta;
    Eigen::Matrix3d second_part = factor_2 * skew_theta_squared;


    
    Eigen::Matrix3d jacobian_matrix = Identity_matrix + first_part + second_part;

    //Translation 
    Eigen::Vector3d translation_part = jacobian_matrix * rho;

    return translation_part;

}

Eigen::Matrix3d compute_rotation_matrix(Eigen::Vector3d  omega ){
//    cv::Mat R;
//    cv::Mat omega_ = (cv::Mat_<double>(3, 1) <<  omega[0],omega[1],omega[2]);
//    cv::Rodrigues(omega_, R);
//    std::cout<< "R -" << R<<std::endl;

    double theta_sq = omega.squaredNorm();
    double theta = std::sqrt(theta_sq);

    Eigen::Matrix3d Identity_matrix = Eigen::Matrix3d::Identity();
    double factor_1 = std::sin(theta) / theta;
    double factor_2 = (1 - std::cos(theta)) /theta_sq;

    Eigen::Matrix3d skew_theta;
    double omega_1 = omega[0];
    double omega_2 = omega[1];
    double omega_3 = omega[2];
    skew_theta << 0 ,     -omega_3 , omega_2,
                omega_3,  0        , -omega_1,
                -omega_2, omega_1  ,  0;
    Eigen::Matrix3d skew_theta_squared = skew_theta * skew_theta;
    Eigen::Matrix3d first_part = factor_1 * skew_theta;
    Eigen::Matrix3d second_part = factor_2 * skew_theta_squared;

    Eigen::Matrix3d rotation_matrix = Identity_matrix + first_part + second_part;
    // std::cout<<"rotation matrix with rodriguz \n"<<rotation_matrix << std::endl;



    /// Compute rotaion matrix with straight formula
    Eigen::Matrix3d rotation_matrix_theta_n;

    double n_1 = omega[0] / theta;
    double n_2 = omega[1] / theta;
    double n_3 = omega[2] / theta;

    double sine_ = std::sin(theta);
    double cos_ = std::cos(theta);

    double index_0_0 = cos_ + (n_1 * n_1 * (1 - cos_));
    double index_0_1 = (n_1 * n_2 * (1- std::cos(theta))) - (n_3 * std::sin(theta));
    double index_0_2 = (n_1 * n_3 * (1- cos_)) + (n_2 * sine_);

    double index_1_0 = (n_1 * n_2 * (1- cos_)) + (n_3 * sine_);
    double index_1_1 = cos_ + (n_2 * n_2 * (1- cos_));
    double index_1_2 = (n_2 * n_3 * (1 - cos_)) - (n_1 * sine_);

    double index_2_0 = (n_1 * n_3 * (1-cos_)) - (n_2 * sine_);
    double index_2_1 = ( n_2 * n_3 * (1 - cos_) ) + (n_1 * sine_);
    double index_2_2 = (cos_) + (n_3 * n_3  * (1-cos_));

    rotation_matrix_theta_n << index_0_0, index_0_1 , index_0_2,
                                index_1_0 ,index_1_1 , index_1_2,
                                index_2_0 , index_2_1, index_2_2;
    //std::cout<<"rotation matrix with theta and n \n"<<rotation_matrix_theta_n << std::endl;
    return rotation_matrix;
}
