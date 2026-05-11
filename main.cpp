#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono> // for performance profiling

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char** argv) {
    if(argc < 2) { // to force the video path to be passed via command line
        cerr << "Usage: " << argv[0] << " <video_path>" << endl;
        return -1;
    }

    VideoCapture cap(argv[1], CAP_FFMPEG); // "/home/diebarr/Progetto_Aspromonte/attempt.mov"

    if(!cap.isOpened()) {
        cerr << "Error: Cannot open video file." << endl;
        return -1;
    }

    // 360° or equirectangular video (input) dimensions
    int video_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int video_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    Mat frame; // empty header
    const int OUTPUT_FACE_DIM = 1024; // output face dimensions
    Mat map_x[3], map_y[3]; // left, front and right matrices of a Cubemap
    for(int i = 0; i < 3; ++i) {
        map_x[i] = Mat(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM, CV_32F);
        map_y[i] = Mat(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM, CV_32F);
    }

    // equirectangular -> cubemap by backward mapping
    for(int face = 0; face < 3; ++face)
        for(int i = 0; i < OUTPUT_FACE_DIM; ++i)
            for(int j = 0; j < OUTPUT_FACE_DIM; ++j) {
                // normalization of the coordinates in [-1, 1]
                float x = (j / (float)OUTPUT_FACE_DIM) * 2.0f - 1.0f;
                float y = (i / (float)OUTPUT_FACE_DIM) * 2.0f - 1.0f;
                float X, Y, Z; // coordinates of the vector needed to move through the faces of the cubemap
                
                switch(face) {
                    case 0: // front face
                        X = x;
                        Y = y;
                        Z = 1.0f;
                        break;
                    case 1: // right face
                        X = 1.0f;
                        Y = y;
                        Z = -x;
                        break;
                    case 2: // left face
                        X = -1.0f;
                        Y = y;
                        Z = x;
                        break;
                }

                // conversion from cartesian vectors to spherical polar coordinates
                float theta = atan2(X, Z); // longitude
                float phi = asin(Y / (sqrt(X * X + Y * Y + Z * Z))); // latitude

                // normalization of the spherical polar coordinates to an equirectangular uv space: [0, 1]
                float u = (theta / (2.0f * CV_PI)) + 0.5f;
                float v = (phi / CV_PI) + 0.5f;

                // population of the matrices
                map_x[face].at<float>(i, j) = u * video_width;
                map_y[face].at<float>(i, j) = v * video_height;
            }

    Mat panoramic_frame(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM * 3, CV_8UC3); // (1024 X 3072, 3 color channels)
    
    // Region of Interest (ROI) headers for each face (1024 x 1024 each)
    Mat left_roi = panoramic_frame(Rect(0, 0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));
    Mat front_roi = panoramic_frame(Rect(OUTPUT_FACE_DIM, 0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));
    Mat right_roi = panoramic_frame(Rect(OUTPUT_FACE_DIM * 2, 0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));

    while (cap.read(frame)) {
#ifdef ENABLE_PROFILING
        auto start = high_resolution_clock::now();
#endif

        remap(frame, front_roi, map_x[0], map_y[0], INTER_LINEAR, BORDER_WRAP);
        remap(frame, right_roi, map_x[1], map_y[1], INTER_LINEAR, BORDER_WRAP);
        remap(frame, left_roi,  map_x[2], map_y[2], INTER_LINEAR, BORDER_WRAP);

#ifdef ENABLE_PROFILING
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        cerr << "Frame time: " << elapsed.count() << " ms\n";
#endif

#ifdef USE_GUI
        imshow("Progetto Aspromonte - Input View", panoramic_frame); // shows the panoramic video
        if (waitKey(1) == 'q') // pauses each frame for 1ms to wait for a 'q' (quit) user input
            break;
#else
        cout.write(reinterpret_cast<const char*>(panoramic_frame.data), panoramic_frame.total() * panoramic_frame.elemSize()); // transforms the frames in raw binary data needed in Python
#endif
    }

    cap.release();

#ifdef USE_GUI
    destroyAllWindows();
#endif
}