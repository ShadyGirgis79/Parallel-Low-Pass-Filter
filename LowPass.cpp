
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <mpi.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

// #define RUN_SEQUENTIAL
// #define RUN_OPENMP
#define RUN_MPI

#ifdef RUN_SEQUENTIAL

int main() {

    // Put the path of the image
    Mat image = imread("D:/Projects/Parallel Low Pass Filter/lena.png", IMREAD_GRAYSCALE);  
    
    // Check if the image was loaded successfully
    if (image.empty()) {
        cout << "Error: Could not open or find the image!" << endl;
        return -1;
    }

    Mat output = image.clone();

    int kSize;

    do {
        cout << "Enter odd kernel size : ";
        cin >> kSize;
    } 
    while (kSize % 2 == 0 || kSize < 3);

    int pad = kSize / 2;
    int kernelSum = kSize * kSize;

    // Calculating the start time
    auto start = high_resolution_clock::now();

    //Looping on the image
    for (int i = pad; i < image.rows - pad; ++i) {
        for (int j = pad; j < image.cols - pad; ++j) {
            int sum = 0;
            for (int ki = -pad; ki <= pad; ++ki) {
                for (int kj = -pad; kj <= pad; ++kj) {
                    sum += image.at<uchar>(i + ki, j + kj);
                }
            }
            output.at<uchar>(i, j) = sum / kernelSum;
        }
    }

    // Calculating the end time
    auto end = high_resolution_clock::now();
    duration<double> duration = end - start;

    // Save the output
    imwrite("output_blurred.png", output);
    cout << "\n" << "Blurring completed successfully!Output saved as 'output_blurred.png'" << endl;

    cout << "\nExecution Time: " << duration.count() << " seconds" << endl;

    return 0;
}

#endif


#ifdef RUN_OPENMP

int main() {

    // Put the path of the image
    Mat image = imread("D:/Projects/Parallel Low Pass Filter/lena.png", IMREAD_GRAYSCALE);

    if (image.empty()) {
        cout << "Error: Could not open or find the image!" << endl;
        return -1;
    }

    Mat output = image.clone();

    int kSize;

    do {
        cout << "Enter odd kernel size : ";
        cin >> kSize;
    } while (kSize % 2 == 0 || kSize < 3);

    int pad = kSize / 2;
    int kernelSum = kSize * kSize;

    // Calculating the start time
    auto start = high_resolution_clock::now();


#pragma omp parallel for

    for (int i = pad; i < image.rows - pad; ++i) {
        for (int j = pad; j < image.cols - pad; ++j) {
            int sum = 0;
            for (int ki = -pad; ki <= pad; ++ki) {
                for (int kj = -pad; kj <= pad; ++kj) {
                    sum += image.at<uchar>(i + ki, j + kj);
                }
            }
            output.at<uchar>(i, j) = sum / kernelSum;
        }
    }

    // Calculating the end time
    auto end = high_resolution_clock::now();
    duration<double> duration = end - start;

    imwrite("blurred_openmp.png", output);
    cout << "\n" << "OpenMP Blurring done! Output saved as 'blurred_openmp.png'" << endl;

    cout << "\nExecution Time: " << duration.count() << " seconds" << endl;

    return 0;
}

#endif


#ifdef RUN_MPI

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image;
    int kSize = 3;  // Default kernel size

    if (rank == 0) {
        do {
            cout << "Enter odd kernel size : ";
            cin >> kSize;

        } while (kSize % 2 == 0 || kSize < 3);
        

        // Load the image in grayscale
        image = imread("D:/Projects/Parallel Low Pass Filter/lena.png", IMREAD_GRAYSCALE);
        if (image.empty()) {
            cout << "Error: Image not loaded!" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    // Broadcast the kernel size to all processes
    MPI_Bcast(&kSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int pad = kSize / 2;
    int kernelSum = kSize * kSize;

    // Broadcast image dimensions
    int rows = 0, cols = 0;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = rows / size;
    int local_data_size = (local_rows + 2 * pad) * cols;
    vector<uchar> local_data(local_data_size);
    vector<uchar> result_data(local_rows * cols);

    // Calculating the start time
    double start_time = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int start = i * local_rows - pad;
            if (start < 0) start = 0;
            int length = local_rows + 2 * pad;
            if (start + length > rows) length = rows - start;

            Mat part = image.rowRange(start, start + length);
            if (i == 0) {
                memcpy(local_data.data(), part.data, length * cols);
            }
            else {
                MPI_Send(part.data, length * cols, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        MPI_Recv(local_data.data(), local_data_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = pad; i < local_rows + pad; ++i) {
        for (int j = pad; j < cols - pad; ++j) {
            int sum = 0;
            for (int ki = -pad; ki <= pad; ++ki) {
                for (int kj = -pad; kj <= pad; ++kj) {
                    int idx = (i + ki) * cols + (j + kj);
                    sum += local_data[idx];
                }
            }
            result_data[(i - pad) * cols + j] = sum / kernelSum;
        }
    }

    vector<uchar> full_result;
    if (rank == 0) full_result.resize(rows * cols);

    MPI_Gather(result_data.data(), local_rows * cols, MPI_UNSIGNED_CHAR,
        full_result.data(), local_rows * cols, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // Calculating the end time
    double end_time = MPI_Wtime();
    double duration = end_time - start_time;

    if (rank == 0) {
        Mat blurred(rows, cols, CV_8UC1, full_result.data());
        imwrite("blurred_mpi.png", blurred);
        cout << "\nMPI Blurring done! Output saved as 'blurred_mpi.png'" << endl;
        cout << "\nExecution Time: " << duration << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}

#endif



