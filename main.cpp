#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>      // for performance profiling
#include <filesystem>  // for filesystem checks

using namespace std;
using namespace cv;
using namespace chrono;
using namespace filesystem;

int main(int argc, char** argv) {

    // ------------------------------------------------------------------ //
    //  INPUT
    // ------------------------------------------------------------------ //

    string video_path;
    const string default_video_path = "attempt.mov";   // fallback local file

    if (argc >= 2)
        video_path = argv[1];                          // command-line argument
    else if (exists(default_video_path)) {
        video_path = default_video_path;
        cerr << "\nAuto-loading local default video: '" << video_path << "'\n";
    } else {
        cerr << "Usage: " << argv[0] << " <video_path>\n";
        return -1;
    }

    // Open the video file using FFMPEG
    VideoCapture cap(video_path, CAP_FFMPEG);
    if (!cap.isOpened()) {
        cerr << "\nERROR: Cannot open video file: '" << video_path << "'\n";
        return -1;
    }

    // ------------------------------------------------------------------ //
    //  EQUIRECTANGULAR -> CUBEMAP
    // ------------------------------------------------------------------ //

    // Input equirectangular dimensions
    int video_width  = cap.get(CAP_PROP_FRAME_WIDTH);
    int video_height = cap.get(CAP_PROP_FRAME_HEIGHT);

    const int OUTPUT_FACE_DIM = 1024;   // each cube face will be 1024x1024

    Mat frame;                          // raw frame from video
    Mat map_x[3], map_y[3];             // remap lookup tables for left, front, right

    for (int i = 0; i < 3; ++i) {
        map_x[i] = Mat(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM, CV_32F);
        map_y[i] = Mat(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM, CV_32F);
    }

    // Pre-compute backward mapping from cubemap pixel coordinates to equirectangular UV
    for (int face = 0; face < 3; ++face)
        for (int i = 0; i < OUTPUT_FACE_DIM; ++i)
            for (int j = 0; j < OUTPUT_FACE_DIM; ++j) {
                // Normalised cubemap coordinates: [-1, 1]
                float x = (j / (float)OUTPUT_FACE_DIM) * 2.0f - 1.0f;
                float y = (i / (float)OUTPUT_FACE_DIM) * 2.0f - 1.0f;

                float X, Y, Z;   // 3D direction vector for the current face pixel

                // Depending on the face, assign the corresponding 3D vector
                switch (face) {
                    case 0: // Left face
                        X = -1.0f;
                        Y =  y;
                        Z =  x;
                        break;
                    case 1: // Front face
                        X =  x;
                        Y =  y;
                        Z =  1.0f;
                        break;
                    case 2: // Right face
                        X =  1.0f;
                        Y =  y;
                        Z = -x;
                        break;
                }

                // Convert to spherical coordinates
                float theta = atan2(X, Z);                                // longitude
                float phi   = asin(Y / sqrt(X*X + Y*Y + Z*Z));            // latitude

                // Normalise to equirectangular UV space [0,1]
                float u = (theta / (2.0f * CV_PI)) + 0.5f;
                float v = (phi   / CV_PI)         + 0.5f;

                // Store the corresponding pixel coordinates in the source equirectangular image
                map_x[face].at<float>(i, j) = u * video_width;
                map_y[face].at<float>(i, j) = v * video_height;
            }

    // ------------------------------------------------------------------ //
    //  OUTPUT IMAGE & FACE REGIONS
    // ------------------------------------------------------------------ //

    // Final cubemap layout: three faces side by side (3072 x 1024)
    Mat panoramic_frame(OUTPUT_FACE_DIM, OUTPUT_FACE_DIM * 3, CV_8UC3);

    // Define regions of interest for each face inside the panoramic frame
    Mat left_roi  = panoramic_frame(Rect(0,                   0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));
    Mat front_roi = panoramic_frame(Rect(OUTPUT_FACE_DIM,     0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));
    Mat right_roi = panoramic_frame(Rect(OUTPUT_FACE_DIM * 2, 0, OUTPUT_FACE_DIM, OUTPUT_FACE_DIM));

    // ------------------------------------------------------------------ //
    //  MAIN VIDEO LOOP
    // ------------------------------------------------------------------ //

    while (cap.read(frame)) {

#ifdef ENABLE_PROFILING
        auto start = high_resolution_clock::now();
#endif
        // Warp equirectangular to each cubemap face using the pre-computed maps
        remap(frame, left_roi,  map_x[0], map_y[0], INTER_LINEAR, BORDER_WRAP);
        remap(frame, front_roi, map_x[1], map_y[1], INTER_LINEAR, BORDER_WRAP);
        remap(frame, right_roi, map_x[2], map_y[2], INTER_LINEAR, BORDER_WRAP);

#ifdef ENABLE_PROFILING
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        cerr << "Frame time: " << elapsed.count() << " ms\n";
#endif

        // Output: either show in a window or stream raw bytes to Python
#ifdef USE_GUI
        imshow("Progetto Aspromonte - Cubemap Preview", panoramic_frame);
        if ((waitKey(1) & 0xFF) == 'q')   // press 'q' to quit
            break;
#else
        // Write raw pixel data to stdout for the Python pipeline
        cout.write(reinterpret_cast<const char*>(panoramic_frame.data), panoramic_frame.total() * panoramic_frame.elemSize());
#endif
    }

    // ------------------------------------------------------------------ //
    //  CLEANUP
    // ------------------------------------------------------------------ //

    cap.release();
#ifdef USE_GUI
    destroyAllWindows();
#endif
}