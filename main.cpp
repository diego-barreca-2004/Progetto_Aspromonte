#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap("/home/diebarr/Progetto_Aspromonte/attempt.mov", CAP_FFMPEG);

    if(!cap.isOpened()) {
        cerr << "Errore nell'apertura del video" << endl;
        return -1;
    }

    Mat frame; // empty header
    // equirectangular video (input) dimensions
    int video_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int video_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    // cubemap face (output) dimensions
    int cubemap_face_width = 1024;
    int cubemap_face_height = 1024;
    // cubemap dimension matrices
    Mat map_x[6];
    Mat map_y[6];
    for(int i = 0; i < 6; ++i) {
        map_x[i] = Mat(cubemap_face_height, cubemap_face_width, CV_32F);
        map_y[i] = Mat(cubemap_face_height, cubemap_face_width, CV_32F);
    }

    // equirectangular -> cubemap by backward mapping
    for(int face = 0; face < 6; ++face)
        for(int i = 0; i < cubemap_face_height; ++i)
            for(int j = 0; j < cubemap_face_width; ++j) {
                // normalization of the coordinates in [-1, 1]
                float x = (j / (float)cubemap_face_width) * 2.0f - 1.0f;
                float y = (i / (float)cubemap_face_height) * 2.0f - 1.0f;
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
                        X = -1;
                        Y = y;
                        Z = x;
                        break;
                    case 3: // back face
                        X = -x;
                        Y = y;
                        Z = -1.0f;
                        break;
                    case 4: // top face
                        X = x;
                        Y = -1.0f;
                        Z = -y;
                        break;
                    case 5: // bottom face
                        X = x;
                        Y = 1.0f;
                        Z = y;
                        break;
                }

                // conversion from cartesian vectors to spherical polar coordinates
                float theta = atan2(X, Z); // longitude
                float phi = asin(Y / (sqrt(X * X + Y * Y + Z * Z))); // latitude

                // normalization of the spherical polar coordinates to an equirectangular uv space: [0, 1]
                float u = (theta / (2 * CV_PI)) + 0.5f;
                float v = (phi / CV_PI) + 0.5f;

                // transformation in the original dimensions
                float pixel_x = u * video_width;
                float pixel_y = v * video_height;

                // population of the matrices
                map_x[face].at<float>(i, j) = pixel_x;
                map_y[face].at<float>(i, j) = pixel_y;
            }

    Mat front_face, right_face, left_face;
    Mat output;

    while (cap.read(frame)) {
        remap(frame, front_face, map_x[0], map_y[0], INTER_LINEAR, BORDER_WRAP);
        remap(frame, right_face, map_x[1], map_y[1], INTER_LINEAR, BORDER_WRAP);
        remap(frame, left_face,  map_x[2], map_y[2], INTER_LINEAR, BORDER_WRAP);
        Mat input[3] = {left_face, front_face, right_face};
        hconcat(input, 3, output);
        // imshow("Left, front and right faces", output); // enable to show the panoramic video
        // imshow("Front face", front_face); // enable to show the front face video
        /* if (waitKey(30) == 'q') // pauses each frame for 30ms -> 1000ms / 16ms = approx. 63FPS
            break; */
        
        cout.write(reinterpret_cast<const char*>(output.data), (output.total() * output.elemSize())); // transforms the frames in raw binary data needed in Python
    }

    cap.release();
    destroyAllWindows();
}