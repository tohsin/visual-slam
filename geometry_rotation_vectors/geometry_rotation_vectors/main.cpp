//
//  main.cpp
//  geometry_rotation_vectors
//
//  Created by Tosin Oseni on 26/06/2022.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    
    // 3D rotation matrix directly using Matrix3d or Matrix3f
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    
    // to represent rotation vector
    // its not  a matrix but we cas use one to represnt it
    // Rotate 45 degrees along the Z axis
    /// roration vector is angle theta  and a vector n
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));
    
    
    cout.precision(3);
    cout << "rotation matrix = \n " << rotation_vector.matrix() << endl; // convert to matrix with matrix()
    
    // to assign directly and do converting to rotation matrix
    rotation_matrix = rotation_vector.toRotationMatrix();
    cout << "rotation matrix = \n " << rotation_matrix << endl;
    
    // coordinatr transfor mation with angle axis
    Eigen::Vector3d v(1, 0 , 0);
    // apply rotation vector to v
    Eigen::Vector3d vector_rotated = rotation_vector * v;
    
    cout << "(1,0,0) after rotation (by angle axis) = " << vector_rotated.transpose() << endl;
    
    
    // or use rotation matrix
    vector_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << vector_rotated.transpose() << endl;
    
    
    // rotation matrix to euler angle
    // seting zyx order ie roll, pitch , yaw
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
    cout << "yaw pich row yall" << euler_angles.transpose() << endl;
    
    
    
    // Here we do the big picture transformatiion with [R T] , [0, 1]
    // Euclidean transformation matrix using Eigen::Isometry
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    // Although called 3d, it is essentially a 4 * 4 matrix
    
    // first we rotate by R
    T.rotate(rotation_vector);
    // then we translate by  a vector
    T.pretranslate(Eigen::Vector3d(1, 3, 4)); // Set the translation vector to (1,3,4)
    cout << "Transform matrix = \n" << T.matrix() << endl;
    
   
    Eigen::Vector3d v_transformed = T * v; // Equivalent to R * v +t
    cout << "v tranformed = " << v_transformed.transpose() << endl;
    
    // For affine and projective transformations, use Eigen::Affine3d and Eigen::Projective3d.
    
    // now we slightly play with quternions
    // obtain quaternion from rotation vector or angle axis
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
    // order of values are (x, y, z, w), and w is the real part, the first three are the imaginary part
    
    // we can also use rotaion matrix
    q = Eigen::Quaterniond(rotation_matrix);
    cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << endl;
    
    // now to perform rotation of vector with quaternion QPQ-1 from book
    // Rotate a vector with a quaternion and use overloaded multiplication
    vector_rotated = q * v; // same as  math is qvq^{−1}
    cout << "(1,0,0) after rotation = " << vector_rotated.transpose() << endl;
    //or to follow the actual math we get
    // expressed by regular vector multiplication, it should be calculated as follows
    // q * P  * q ^ -1
    cout << "should be equal to " << (q * Eigen::Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs
              ().transpose() << endl;

    
    // Rotation matrix ( 3 × 3 ): Eigen::Matrix3d.
//    Rotation vector ( 3 × 1 ): Eigen::AngleAxisd.
//    Euler angle ( 3 × 1 ): Eigen::Vector3d.
//    Quaternion ( 4 × 1 ): Eigen::Quaterniond.
//Euclidean transformation matrix ( 4 × 4 ): Eigen::Isometry3d.
    //• Aﬀine transform ( 4 × ): Eigen::Aﬀine3d.
//    Perspective transformation ( 4 × 4 ): Eigen::Projective3d.
    return 0;
}
