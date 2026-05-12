#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath> // not strictly necessary as the math functions used are already present in <opencv2/opencv.hpp>
#include <chrono> // for performance profiling
#include <filesystem> // for input video checking

using namespace std;
using namespace cv;
using namespace chrono;
using namespace filesystem;

int main(int argc, char** argv) {
    // --- INPUTS --- //

    string video_path;
    string default_video_path = "attempt.mov"; // local video input

    if(argc >= 2) // to use the video path passed via command line
        video_path = argv[1];
    else if(exists(default_video_path)) { // to use the local video input
            video_path = default_video_path;
            cerr << "\nAuto-loading local default video: '" << video_path << '\'' << endl;
    }
    else { // to force to pass the video path via command line
        cerr << "Correct usage is: " << argv[0] << " <video_path>" << endl;
        return -1;
    }

    VideoCapture cap(video_path, CAP_FFMPEG);

    if(!cap.isOpened()) {
        cerr << "\nERROR! Cannot open video file at: '" << video_path << '\'' << endl;
        return -1;
    }

    // 360° or equirectangular video (input) dimensions
    int video_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int video_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    const int OUTPUT_FACE_DIM = 1024; // output face dimensions

    Mat frame; // empty header
    Mat map_x[3], map_y[3]; // left, front and right matrices of a Cubemap

    for(int i = 0; i < 3; ++i) {
        map_x[i] = Mat(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM, CV_32F);
        map_y[i] = Mat(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM, CV_32F);
    }

    // --- EQUIRECTANGULAR -> CUBEMAP (BY BACKWARD MAPPING) --- //

    for(int face = 0; face < 3; ++face)
        for(int i = 0; i < OUTPUT_FACE_DIM; ++i)
            for(int j = 0; j < OUTPUT_FACE_DIM; ++j) {
                float X, Y, Z; // vector coordinates needed to move through the faces of the Cubemap
                // Coordinates normalization to: [-1, 1]
                float x = (j / (float)OUTPUT_FACE_DIM) * 2.0f - 1.0f;
                float y = (i / (float)OUTPUT_FACE_DIM) * 2.0f - 1.0f;
                
                switch(face) {
                    case 0: // left face
                        X = -1.0f;
                        Y = y;
                        Z = x;
                        break;
                    case 1: // front face
                        X = x;
                        Y = y;
                        Z = 1.0f;
                        break;
                    case 2: // right face
                        X = 1.0f;
                        Y = y;
                        Z = -x;
                        break;
                }

                // Cartesian -> Spherical polar coordinates
                float theta = atan2(X, Z); // longitude
                float phi = asin(Y / (sqrt(X * X + Y * Y + Z * Z))); // latitude

                // Spherical polar coordinates normalization to an Equirectangular uv space: [0, 1]
                float u = (theta / (2.0f * CV_PI)) + 0.5f;
                float v = (phi / CV_PI) + 0.5f;

                // Matrices population
                map_x[face].at<float>(i, j) = u * video_width;
                map_y[face].at<float>(i, j) = v * video_height;
            }

    Mat panoramic_frame(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM * 3, CV_8UC3); // (3072 x 1024, 3 color channels)
    
    // Region of Interest (ROI) headers for each face (1024 x 1024 each)
    Mat left_roi = panoramic_frame(Rect(0, 0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));
    Mat front_roi = panoramic_frame(Rect(OUTPUT_FACE_DIM, 0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));
    Mat right_roi = panoramic_frame(Rect(OUTPUT_FACE_DIM * 2, 0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));

    // --- OUTPUTS --- //

    while (cap.read(frame)) {
#ifdef ENABLE_PROFILING // to acquire information on the time (in ms) taken for each frame
        auto start = high_resolution_clock::now();
#endif

        // Panoramic matrix population
        remap(frame, left_roi,  map_x[0], map_y[0], INTER_LINEAR, BORDER_WRAP);        
        remap(frame, front_roi, map_x[1], map_y[1], INTER_LINEAR, BORDER_WRAP);
        remap(frame, right_roi, map_x[2], map_y[2], INTER_LINEAR, BORDER_WRAP);

#ifdef ENABLE_PROFILING
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        cerr << "Frame time: " << elapsed.count() << " ms\n";
#endif

#ifdef USE_GUI
        imshow("Progetto Aspromonte - Input View", panoramic_frame); // shows the panoramic video
        if ((waitKey(1) &  0xFF) == 'q') // waits each ms for a user input ('q'). 0xFF is needed to get each time the real 'q' ASCII value (113)
            break;
#else
        cout.write(reinterpret_cast<const char*>(panoramic_frame.data), panoramic_frame.total() * panoramic_frame.elemSize()); // transforms the frames in raw binary data needed in .py
#endif
    }

    cap.release();

#ifdef USE_GUI
    destroyAllWindows();
#endif
}