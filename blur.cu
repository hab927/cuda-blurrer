#include <cuda_runtime_api.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <chrono>

#define _USE_MATH_DEFINES
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <math.h>
#include "stb_image.h"
#include "stb_image_write.h"

typedef std::vector<std::vector<double>> doubleMatrix;

enum BlurType {
	GAUSSIAN,
	BOX
};

struct Image_Data {
	int w;
	int h;
	int c;
	int total_pixels;
};

static double sample_gaussian(double x, double sigma) {
	// x is the distance from the origin (one axis in this case)
	// sigma is standard deviation (controls the "width", the larger sigma is, the more "spread out" it will be)
	// best explanation i can give
	double denominator = sqrt(2.0 * M_PI * pow(sigma, 2.0));
	double exponent = -1.0 * ((pow(x, 2.0)) / (2.0 * pow(sigma, 2.0)));
	return (1.0 / denominator) * pow(M_E, exponent);
}

static void box_filter_blur_cpu(int row, int col, int radius, struct Image_Data props, unsigned char* source_data, unsigned char* final_data) {
	int w = props.w;
	int h = props.h;
	int c = props.c;

	int pixel_index = (row * w * c) + col;		// this is given ACCOUNTING FOR CHANNELS.

	// bounds for radius blur
	int lower_bound = std::max(row - radius, 0); // ensures 0 is chosen if it goes past top row
	int upper_bound = std::min(row + radius, h - 1); // ensures bottom row is chosen if it goes past
	int left_bound = std::max(col / c - radius, 0); // ensures left column is chosen if it goes past
	int right_bound = std::min(col / c + radius, w - 1); // ensures right column is chosen if it goes past

	int sum = 0;
	int total_squares = (upper_bound - lower_bound + 1) * (right_bound - left_bound + 1);

	for (int channel = 0; channel < c; channel++) {		// we need to do this averaging thing for every channel
		sum = 0;
		for (int y = lower_bound; y <= upper_bound; y++) {
			for (int x = left_bound; x <= right_bound; x++) {
				int pixel_color_data = source_data[(y * w + x) * c + channel];
				sum += pixel_color_data;
			}
		}
		printf("#");
		final_data[pixel_index + channel] = sum / total_squares;
	}
}

// same as above function except minor changes:
// - using max() and min() instead of std::max() and std::min() because cuda complains
// - using gpu block and thread ids to figure out the row and column (thankfully very simple!)
// - functions for larger images and is optimized for smaller ones
__global__ void box_filter_blur_gpu(int radius, struct Image_Data props, unsigned char* source_data, unsigned char* final_data, int start_index = -1) {
	// master box blur function
	// if there is a start index specified, it will split the image into chunks
	// if the image is small enough, the GPU will do all pixels at once.
	int w = props.w;
	int h = props.h;
	int c = props.c;
	int total_pixels = props.total_pixels;

	int row, col, pixel_index;

	if (start_index == -1) {
		row = blockIdx.x;
		col = threadIdx.x * c;
		pixel_index = (row * w * c) + col;		// accounts for channels
	}
	else {
		pixel_index = start_index + (blockIdx.x * blockDim.x + threadIdx.x); // this time it will not account for channels for simplicity.
		if (pixel_index > total_pixels - 1) {
			// no overflow
			return;
		}
		row = pixel_index / w;
		col = pixel_index % w * c;
	}

	int lower_bound = max(row - radius, 0);
	int upper_bound = min(row + radius, h - 1);
	int left_bound = max(col / c - radius, 0);
	int right_bound = min(col / c + radius, w - 1);

	int total_squares = (upper_bound - lower_bound + 1) * (right_bound - left_bound + 1);

	for (int channel = 0; channel < c; channel++) {
		int sum = 0;
		for (int y = lower_bound; y <= upper_bound; y++) {
			for (int x = left_bound; x <= right_bound; x++) {
				int pixel_color_data = source_data[(y * w + x) * c + channel];
				sum += pixel_color_data;
			}
		}
		final_data[(pixel_index * c) + channel] = sum / total_squares;
	}
}

__global__ void gaussian_blur(int start_index, int matrix_size, double* gaussian_matrix, struct Image_Data props, unsigned char* source_data, unsigned char* final_data) {
	// gaussian blur function
	// this is similar to the box blur function save for a few differences:
	// instead of taking the average of all neighboring pixels, it samples from the provided Gaussian matrix
	// and uses these as "weights" for every surrounding pixel when summed together.
	// this has to be done per channel like normal, so the nested for loop looks the same.
	// the only thing that changes is the math within.
	int w = props.w;
	int h = props.h;
	int c = props.c;
	int total_pixels = props.total_pixels;

	int pixel_index = start_index + (blockIdx.x * blockDim.x + threadIdx.x); // accounts not for channels
	if (pixel_index > total_pixels - 1) {
		// no overflow
		return;
	}
	int row = pixel_index / w;
	int col = pixel_index % w;
	int radius = (matrix_size - 1) / 2;
	int m_length = matrix_size * matrix_size;

	// new bound technology!
	// the non-testing bounds are the true bounds of the image
	// however, the program will use the testing bounds to "check", and allow for overflow
	// if the given dimension is outside any of the bounds, it will just continue the loop
	// that way, it is effectively adding 0 to all of the channels as if it was a blank pixel
	// this gives the fuzzy edges characteristic in gaussian blur

	for (int channel = 0; channel < c; channel++) {
		double sum = 0;
		for (int y = -radius; y <= radius; y++) {
			if (row + y < 0 || row + y >= h) {
				continue;
			}
			for (int x = -radius; x <= radius; x++) {
				if (col + x < 0 || col + x >= w) {
					continue;
				}
				// our pixel is within the range, so multiply it by the corresponding
				// value in the gaussian matrix
				sum +=	source_data[
							((row + y) * w + (col + x)) 
							* c + 
							channel
						] 
						* gaussian_matrix[
							(y + radius)
							* matrix_size 
							+ (x + radius)
						];
				//int pixel_color_data = source_data[(y * w + x) * c + channel];
				//sum += pixel_color_data;
			}
		}
		final_data[(pixel_index * c) + channel] = sum;
	}
}

static void write_image(std::string original_filename, BlurType type, int width, int height, int channels, unsigned char* pixel_data) {

	std::string type_string;
	switch (type) {
		case GAUSSIAN:
			type_string = "gaussian";
			break;
		case BOX:
			type_string = "box";
			break;
		default:
			type_string = "unknown";
			break;
	}

	std::string pathname = "../images/" + original_filename + "_" + type_string + ".png";
	const char* filename = pathname.c_str();
	int success = stbi_write_png(filename, width, height, channels, pixel_data, width * channels);

	if (success) {
		std::cout << "Image successfully saved to " << filename << std::endl;
	}
	else {
		std::cerr << "Failed to save image." << std::endl;
	}
}

static void help_message(std::string error_msg = "") {
	std::cout << "This program is meant to blur images using NVIDIA's CUDA Toolkit.\nDefault behavior is using box blur with radius 3.\n\n" 
		<< "Program Usage:\n\t./blur.exe [options]\n\n"
		<< "Options:\n\t"
			"-c : use CPU for rendering (slower)\n\t"
			"-r <int> : choose blur radius for box blur (int, default 3)\n\t"
			"-t : enable timing\n\t"
			"-g : use gaussian blur (requires CUDA)\n\t"
			"-s : specify standard deviation for gaussian blur (double, default 1.0)\n\t"
		<< std::endl;
}

int main(int argc, char* argv[]) {
	// default values
	int blur_radius = 3;
	double std_dev = 1.0; // for Gaussian blur

	// multithreading
	int block_size = 512;		// max 1024
	int total_blocks = 256;		// max 1024

	bool timed = false;
	bool cpu_mode = false;
	bool debug_mode = false;
	bool use_box_blur = true;
	bool use_gaussian_blur = false;
	enum BlurType bt = BOX;

	// extract filename from the first argument argv[1]
	std::string original_filename;
	std::string image_url = argv[1];
	std::cout << image_url;
	size_t index = image_url.rfind('\\');
	if (index == std::string::npos) {
		std::cout << "no slash found\n";
		original_filename = "blank";
	}
	std::string url_piece = image_url.substr(index);
	original_filename = url_piece.substr(0, url_piece.length() - 4);

	for (int i = 0; i < argc; i++) {
		std::string arg = argv[i];

		if (arg[0] == '-' && arg.length() == 2) {
			switch (arg[1]) {
				case 'r':
					if (argv[i + 1] != NULL) {
						long r = strtol(argv[i + 1], nullptr, 10);
						if (r != 0L) {
							blur_radius = r;
						}
					}
					break;
				case 't':
					timed = true;
					break;
				case 'c':
					cpu_mode = true;
					break;
				case 's':
					if (argv[i + 1] != NULL) {
						double s = strtod(argv[i + 1], nullptr);
						if (s != 0.0) {
							std_dev = s;
						}
					}
					break;
				//case 'd':
				//	debug_mode = true;
				//	break;
				case 'g':
					use_box_blur = false;
					use_gaussian_blur = true;
					bt = GAUSSIAN;
					break;
				default:
					help_message();
					break;
			}
		}
	}

	if (argc < 2 && !debug_mode) {
		help_message();
		return 1;
	}

	// DEBUG ZONE (nefarious testing purposes)

	// DEBUG ZONE

	if (debug_mode) {
		std::cout << "Exiting program early for debugging purposes.\n" << std::endl;
		return 1;
	}

	int w, h, c;	// image properties

	char* filename = argv[1];

	// CPU (host) variables that need to be copied
	unsigned char* source_data = nullptr;
	unsigned char* final_data = nullptr;

	// GPU (device) variables (following convention from nvidia's cuda guide https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
	unsigned char* dev_source_data = nullptr;
	unsigned char* dev_final_data = nullptr;
	double* dev_gaussian_matrix;

	source_data = stbi_load(filename, &w, &h, &c, 0);
	
	//for (int i = 0; i < w * h * c; i++) {
	//	std::cout << static_cast<int>(source_data[i]) << ",";
	//}

	Image_Data props = { w, h, c, w*h };

	//allocation of memory into cpu and gpu
	int image_size = sizeof(unsigned char) * w * h * c;

	final_data = new unsigned char[w * h * c];
	cudaMalloc(&dev_source_data, image_size);
	cudaMalloc(&dev_final_data, image_size);

	cudaMemcpy(dev_source_data, source_data, image_size, cudaMemcpyDefault);
	cudaMemset(dev_final_data, 0, image_size);

	if (source_data == NULL) {
		std::cout << stbi_failure_reason();
		return 1;
	}
	else {
		std::cout << "\nHardware Used: " << (cpu_mode ? "CPU" : "GPU") << "\n-------\n"
					<< "Width: " << w << "\n" 
					<< "Height: " << h << "\n"
					<< "Channels: " << c << "\n\n"
					<< std::endl;
	}

	if (use_box_blur) {
		std::cout << "Using box blur with radius " << blur_radius << "...\n";
	}
	if (use_gaussian_blur) {
		std::cout << "Using gaussian blur with standard deviation " << std_dev << "...\n";
	}

	const auto start = std::chrono::steady_clock::now();

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// box filter blurring
	// this is the default option
	if (use_box_blur) {
		if (cpu_mode) {
			for (int row = 0; row < h; row++) {
				for (int col = 0; col < w * c; col += c) {	// to account for channels
					box_filter_blur_cpu(row, col, blur_radius, props, source_data, final_data);
				}
			}
		}
		else if ((h < 1024) && (w < 1024)) {
			box_filter_blur_gpu<<< h, w, 0, stream >>>(blur_radius, props, dev_source_data, dev_final_data);
		}
		else {
			// for this we have to split it into chunks
			// split image into chunks of size k by dividing total pixels by chunk size (1 thread per pixel so block size * thread count)
			// every iteration, increment a cursor value to the last pixel
			// pass this cursor value to the function so it knows where to start
			int increment = block_size * total_blocks;

			for (int k = 0; k < w * h; k += increment) { // k is the "start index" mentioned above
				box_filter_blur_gpu<<< total_blocks, block_size, 0, stream >>>(blur_radius, props, dev_source_data, dev_final_data, k);
			}
		}
	}

	// gaussian blurring
	// no i am NOT going to let you use the CPU for this one, it's way too expensive

	if (use_gaussian_blur) {
		// produce the gaussian matrix according to the standard deviation given above
	
		// determine the size of the matrix by limiting each direction from the origin to be 3*sigma
		int matrix_radius = static_cast<int>(ceil(3 * std_dev));
		doubleMatrix gaussian_matrix(2 * matrix_radius + 1, std::vector<double>(2 * matrix_radius + 1, 0));

		// two passes: first axis gaussian distribution, then second axis (vertical) multiplying with values from the 1D pass
		// pass 1 (horizontal)
		for (int y = -matrix_radius; y <= matrix_radius; y++) {
			for (int x = -matrix_radius; x <= matrix_radius; x++) {
				gaussian_matrix[y + matrix_radius][x + matrix_radius] = sample_gaussian(x, std_dev);
			}
		}
		// pass 2 (vertical)
		for (int y = -matrix_radius; y <= matrix_radius; y++) {
			for (int x = -matrix_radius; x <= matrix_radius; x++) {
				gaussian_matrix[y + matrix_radius][x + matrix_radius] = sample_gaussian(y, std_dev) * gaussian_matrix[y + matrix_radius][x + matrix_radius];
			}
		}

		// normalize the matrix
		double sum = 0;
		int terms = pow(2 * matrix_radius + 1, 2);
		int m_size = gaussian_matrix.size();
		for (int y = 0; y < m_size; y++) {
			for (int x = 0; x < m_size; x++) {
				sum += gaussian_matrix[y][x];
			}
		}
		double factor = 1.0 / sum;
		for (int y = 0; y < m_size; y++) {
			for (int x = 0; x < m_size; x++) {
				gaussian_matrix[y][x] *= factor;
			}
		}

		int g_matrix_bytes = sizeof(double) * pow(gaussian_matrix.size(), 2);
		cudaMalloc(&dev_gaussian_matrix, g_matrix_bytes);

		double* flattened_matrix = new double[m_size * m_size] {};

		for (int y = 0; y < m_size; y++) {
			for (int x = 0; x < m_size; x++) {
				flattened_matrix[y * m_size + x] = gaussian_matrix[y][x];
			}
		}

		//for (int i = 0; i < m_size * m_size; i++) {
		//	std::cout << flattened_matrix[i];
		//}

		cudaMemcpy(dev_gaussian_matrix, flattened_matrix, g_matrix_bytes, cudaMemcpyHostToDevice);

		//for (int i = 0; i < m_size * m_size; i++) {
		//	std::cout << dev_gaussian_matrix[i];
		//}

		int increment = block_size * total_blocks;
		// flatten the 2D gaussian matrix into a 1D array to make it easier in the kernel
		for (int k = 0; k < w * h; k += increment) { // k is the "start index" mentioned above
			gaussian_blur<<< total_blocks, block_size, 0, stream >>>(k, m_size, dev_gaussian_matrix, props, dev_source_data, dev_final_data);
		}

	}

	cudaStreamSynchronize(stream);
	cudaMemcpy(final_data, dev_final_data, image_size, cudaMemcpyDefault);

	if (timed) {
		const auto finish{ std::chrono::steady_clock::now() };
		const std::chrono::duration<double> elapsed_seconds{ finish - start };
		std::cout << "Blur time: " << elapsed_seconds.count() << " seconds" << "\n\n";
	}

	// image data is left-to-right, top-to-bottom, 4 channels (R,G,B,A), w * h pixels
	// this is really slow right now for big images, so find a way to optimize later
	write_image(original_filename, bt, w, h, c, final_data);

	// freeing
	cudaStreamDestroy(stream);
	cudaFreeHost(source_data);
	cudaFreeHost(final_data);
	cudaFree(dev_source_data);
	cudaFree(dev_final_data);
	stbi_image_free(source_data);

	return 0;
}