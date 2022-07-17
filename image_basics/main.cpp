#include <iostream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

// test if open cv works
//int main() {
//    cv::Mat srcImage = cv::imread("/Users/emma/dev/visual-slam/image_basics/ubuntu.png");
//    cv::imshow("[img]", srcImage);
//    cv::waitKey(0);
//    return 0;
//}
std::string image_dir = "/Users/emma/dev/visual-slam/image_basics/ubuntu.png";
int main(int argc, char **argv) {
    cv::Mat image = cv::imread(image_dir);

    if (image.data == nullptr){ // check if ffile exist
        std::cerr<< "file not found" << std::endl;
        return 0;
    }

    std::cout << "Image Cols " << image.cols << " rows " << image.rows << " channels " << image.channels()<<std::endl;

//    cv::imshow("[img]", image);
//    cv::waitKey(0);


    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        // we need grayscale image or RGB image
        std::cout << "image type incorrect." << std::endl;
        return 0;
    }

    std::chrono::steady_clock::time_point time_1 = std::chrono::steady_clock::now();
    for ( size_t y =0 ; y<image.rows ; y++ ){
        // use cv::Mat::ptr to get the pointer of each row
        unsigned char *row_ptr = image.ptr<unsigned char>(y);
        for(size_t x = 0 ; x<image.cols ; x++){
            // data_ptr the pointer to (x,y)
            unsigned char *data_ptr = &row_ptr[x * image.channels()];

            // visit every pixel in channel
            for (int c=0;c<image.channels();c++){
                unsigned char data = data_ptr[c];
                // data should be pixel of I(x,y) in c−th channel
            }
        }
    }
    // time clock thingy
    std::chrono::steady_clock::time_point time_2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast< std::chrono::duration < double > > (time_2 - time_1);

    std::cout << "time used: " << time_used.count() << " seconds." << std::endl;

    // copying an image in open cv with = doesn't work lets try without  copy method
    cv::Mat image_copy = image;
    image_copy(cv::Rect(0, 0, 100, 100)).setTo(0); // attempt  to set top 100 * 100 to zero or black

    cv::imshow("image with  copy zero", image);
    cv::waitKey(0);

    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    // We are not going to copy the OpenCV's documentation here
    // please take a look at it for other image operations like clipping, rotating and scaling.
    cv::destroyAllWindows();
    return 0;
}

