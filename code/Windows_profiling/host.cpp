
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <string> 
#include <fstream> 
#include <malloc.h> 
#include <windows.h>
#include <math.h>
#include <random>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "C:\EE524\project\VS\header_files\stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "C:\EE524\project\VS\header_files\stb_image_write.h"

#include "C:\EE524\project\VS\header_files\read_source.h"

#include <CL/cl.h> 

#define SEPARATOR       ("----------------------------------------------------------------------\n") 
#define INTEL_PLATFORM  "Intel(R) OpenCL" 
#define PI 3.14f
#define SIMILARITY_WINDOW 7U
#define SEARCH_WINDOW 21U
#define MAX_IERATIONS 2
using namespace std;

// get platform id of Intel OpenCL platform 
cl_platform_id get_intel_platform(const string platformName);
//unsigned char *noisy_image = NULL;
// read the kernel source code from a given file name 
//char* read_source(const char *file_name, size_t* file_size); 

// print the build log in case of failure 
void build_fail_log(cl_program, cl_device_id);

void generateGreyscaleImage(unsigned char *input_img, unsigned char *greyscale_image, int cols, int rows, int channels, cl_sampler *sampler);
//generate noisy image from the input image
void generateNoisyImage(unsigned char *input_Img, unsigned char *noisy_image, int cols, int rows, int channels, cl_sampler *sampler, float *gaussianRandomDist);

void denoise(unsigned char *noisyImage, int cols, int rows, int channels, cl_sampler *sampler, float stdDeviation, float filterParameter);

void method_noise(unsigned char *input_image, unsigned char *denoised_image, int cols, int rows, int channels,
	cl_sampler *sampler);


int main(int argc, char** argv)
{
	cl_int err;                             // error code returned from api calls 
	cl_platform_id   platform = NULL;   // platform id 
	cl_device_id     device_id = NULL;   // compute device id  
	cl_context       context = NULL;   // compute context 
	cl_program       program1 = NULL;   // compute program 
	cl_program       program3 = NULL;

	string CLFileName1;
	string CLKernelName1;
	string CLFileName3;
	string CLKernelName3;
	string outFilename;
	string inFile;
	string inPath = "C:/EE524/project/VS/input/";
	string outPath = "C:/EE524/project/VS/output/";


	float* hInputImg = NULL;
	float* hOututImg = NULL;

	float theta = 45.0f;  // rotation angle, degrees.

	int imgRows = 0;
	int imgCols = 0;
	int imgChannels;

	int runtype = 0;

	unsigned char *imgdata = NULL;

	unsigned char *hGreyScaleImage = NULL;
	unsigned char *hNoisyImage = NULL;
	unsigned char *hDenoisedImage = NULL;


	int dim0 = 2;
	size_t local0[] = { 8, 8, 0 };
	size_t global0[] = { 0, 0, 0 };


	if (argc < 2)
	{
		printf("ERROR Invalid usage of program. Requires at least 2 command line parameter inputs! Exiting...\n");
		return 0;
	}
	else
	{
		if (argc != 3)
		{
			printf("ERROR Invalid usage of program. Requires 3 command line inputs: runtype theta infile outfile! Exiting...\n");
			return 0;
		}
		//selecting kernel
		CLFileName1 = "greyscale.cl";
		CLKernelName1 = "convert_to_greyscale";
		printf("Executing Kernel File: %s\n", CLFileName1.c_str());
		CLFileName3 = "denoise.cl";
		CLKernelName3 = "denoise_algorithm";
		printf("Executing Kernel File: %s\n", CLFileName3.c_str());

		dim0 = 2;
		inFile = argv[1];
		outFilename = argv[2];

		imgdata = stbi_load(inPath.append(inFile).c_str(), &imgCols, &imgRows, &imgChannels, 4);
		imgChannels = 4;
		//since the image is always converted to 32 bit number of channels is 4
		hGreyScaleImage = (unsigned char*)malloc(imgRows*imgCols * imgChannels * sizeof(unsigned char));
		printf("Convert to Greyscale: Kernel File: %s, Kernel Function: %s, infile: %s, outputfile: %s\n", CLFileName1.c_str(), CLKernelName1.c_str(), inFile.c_str(), "greyscale_image");
		hDenoisedImage = (unsigned char*)malloc(imgRows*imgCols * imgChannels * sizeof(unsigned char));
		printf("Denoise: Kernel File: %s, Kernel Function: %s, infile: %s, outputfile: %s\n", CLFileName3.c_str(), CLKernelName3.c_str(), "noisy_image", outFilename.c_str());

		global0[0] = imgCols;
		global0[1] = imgRows;
		global0[2] = 0;// #partitions in dim * # WIs in dim
	}

	// get Intel OpenCL platform 
	platform = get_intel_platform(INTEL_PLATFORM);

	if (NULL == platform)
	{
		printf("Error: failed to found Intel platform...\n");
		return EXIT_FAILURE;
	}

	// Getting the compute device for the processor graphic (GPU) on our platform by function 
	printf("Selected device: GPU\n");
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	char *deviceName = new char[1024];
	err |= clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, deviceName, NULL);
	printf("Device name: %s\n", deviceName);
	if (CL_SUCCESS != err || NULL == device_id)
	{
		printf("Error: Failed to get device on this platform!\n");
		return EXIT_FAILURE;
	}

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (CL_SUCCESS != err || NULL == context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	size_t file_size1;
	char * kernel_source1 = read_source(CLFileName1.c_str(), &file_size1);
	if (NULL == kernel_source1)
	{
		printf("Error: Failed to read kernel source code from file name: %s!\n", CLFileName1.c_str());
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	size_t file_size3;
	char * kernel_source3 = read_source(CLFileName3.c_str(), &file_size3);
	if (NULL == kernel_source3)
	{
		printf("Error: Failed to read kernel source code from file name: %s!\n", CLFileName3.c_str());
		clReleaseContext(context);
		return EXIT_FAILURE;
	}


	program1 = clCreateProgramWithSource(context, 1, (const char **)&kernel_source1, NULL, &err);
	if (CL_SUCCESS != err || NULL == program1)
	{
		printf("Error: Failed to create compute program1!\n");
		clReleaseContext(context);
		return EXIT_FAILURE;
	}
	program3 = clCreateProgramWithSource(context, 1, (const char **)&kernel_source3, NULL, &err);
	if (CL_SUCCESS != err || NULL == program3)
	{
		printf("Error: Failed to create compute program3!\n");
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	// Build the executable program object
	//printf("\nBuild and Compile the program executable\n");
	//	-cl - mad - enable - cl - fast - relaxed - math
	err = clBuildProgram(program1, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
	err |= clBuildProgram(program3, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to build program executable!\n");
		build_fail_log(program1, device_id);
		clReleaseProgram(program1);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	// creates a command-queue on a specific device. context must be a valid OpenCL context. 
	//printf("\nCreating a command queue with properties\n");

	cl_command_queue commands0 = clCreateCommandQueueWithProperties(context, device_id, NULL, &err); // NO Profiling
	if (CL_SUCCESS != err || NULL == commands0)
	{
		printf("Error: Failed to create a Host command queue!\n");
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	// Create the compute kernel object in the program we wish to run 
#define NUM_KERNELS 3

	cl_kernel denoise_kernels[NUM_KERNELS] = { 0 };
	denoise_kernels[0] = clCreateKernel(program1, CLKernelName1.c_str(), &err);
	if (CL_SUCCESS != err || NULL == denoise_kernels[0])
	{
		printf("Error %d: Failed to create compute kernel0!\n", err);
		clReleaseProgram(program1);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	denoise_kernels[2] = clCreateKernel(program3, CLKernelName3.c_str(), &err);
	if (CL_SUCCESS != err || NULL == denoise_kernels[2])
	{
		printf("Error %d: Failed to create compute kernel2!\n", err);
		clReleaseProgram(program3);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}
	//define sampler
	cl_sampler sampler;
	cl_addressing_mode addressing_mode = CL_ADDRESS_CLAMP_TO_EDGE;
	cl_filter_mode filter_mode = CL_FILTER_LINEAR;
	const cl_sampler_properties sampler_properties[] = { CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE, CL_SAMPLER_ADDRESSING_MODE, addressing_mode , CL_SAMPLER_FILTER_MODE, filter_mode, 0 };
	sampler = clCreateSamplerWithProperties(context, sampler_properties, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR %d : clCreateSamplerWithProperties not executed successfully\n", err);
	}

	cl_image_desc desc;
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = imgCols;
	desc.image_height = imgRows;
	desc.image_depth = 0;
	desc.image_array_size = 0;
	desc.image_slice_pitch = 0;
	desc.image_row_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
	desc.buffer = NULL;
	cl_image_format imgfmt;
	imgfmt.image_channel_order = CL_RGBA;
	imgfmt.image_channel_data_type = CL_UNSIGNED_INT8;

	cl_mem inputImg1 = clCreateImage(context, CL_MEM_READ_ONLY, &imgfmt, &desc, NULL, &err);
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to create input image object! %d\n", err);
		clReleaseProgram(program1);
		clReleaseProgram(program3);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	cl_mem inputImg3 = clCreateImage(context, CL_MEM_READ_ONLY, &imgfmt, &desc, NULL, &err);
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to create input image object! %d\n", err);
		clReleaseProgram(program1);
		clReleaseProgram(program3);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	cl_mem outputImg1 = clCreateImage(context, CL_MEM_WRITE_ONLY, &imgfmt, &desc, NULL, &err);
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to create output image object!\n");
		clReleaseProgram(program1);
		clReleaseProgram(program3);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	cl_mem outputImg3 = clCreateImage(context, CL_MEM_WRITE_ONLY, &imgfmt, &desc, NULL, &err);
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to create output image object!\n");
		clReleaseProgram(program1);
		clReleaseProgram(program3);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}
	// copy host data to device

	size_t origin[3] = { 0,0,0 };  // offset within image to copy from
	size_t region[3] = { imgCols, imgRows, 1 }; // elements per dim
	err = clEnqueueWriteImage(commands0, inputImg1, CL_TRUE, origin, region, 0, 0, imgdata, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		printf("Error %d: clEnqueueWriteImage Failed!\n", err);
		clReleaseProgram(program1);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}

	//Run kernel1 to generate the greyscale image from th input image
	// Setting the arguments to our compute kernel in order to execute it. 
	err = clSetKernelArg(denoise_kernels[0], 0, sizeof(cl_mem), &inputImg1);
	//printf("Setting argument number 0 in convert_to_greyscale kernel\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 0 in convert_to_greyscale kernel!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[0], 1, sizeof(cl_mem), &outputImg1);
	//printf("Setting argument number 1 in convert_to_greyscale kernel\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 1 in convert_to_greyscale kernel!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[0], 2, sizeof(int), &imgCols);
	//printf("Setting argument number 2 in convert_to_greyscale kernel\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 2 in convert_to_greyscale kernel!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[0], 3, sizeof(int), &imgRows);
	//printf("Setting argument number 3 in convert_to_greyscale kernel\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 3 in convert_to_greyscale kernel!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[0], 4, sizeof(cl_sampler), &sampler);
	//printf("Setting argument number 4\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 4 in convert_to_greyscale kernel!\n");
		return EXIT_FAILURE;
	}

	//getting the profiling info
	cl_ulong ctr_start = 0, ctr_end = 0, freq = 0;
	cl_ulong sum_execution_time = 0;

	//calculating profiling info for 100 iterations
	cl_ulong kernel_execution_time[MAX_IERATIONS];
	printf("\nconvert_to_greyscale kernel performance evaluation.");
	for (int rpts = 0; rpts < MAX_IERATIONS; rpts++)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_start));
		err = clEnqueueNDRangeKernel(commands0, denoise_kernels[0], dim0, NULL, global0, local0, 0, NULL, NULL);

		if (CL_SUCCESS != err)
		{
			printf("Error %d: Failed to execute kernel!\n", err);
			clReleaseKernel(denoise_kernels[0]);
			clReleaseProgram(program1);
			clReleaseCommandQueue(commands0);
			clReleaseContext(context);
			return EXIT_FAILURE;
		}
		err = clFinish(commands0);
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_end));
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			clReleaseKernel(denoise_kernels[0]);
			clReleaseProgram(program1);
			clReleaseCommandQueue(commands0);
			clReleaseContext(context);
			return EXIT_FAILURE;
		}
		//To get more precise data we convert to micro s before dividing by ticks-per-second.
		cl_ulong temp = (ctr_end - ctr_start) * 1000000;
		kernel_execution_time[rpts] = (temp * (1.0 / freq));
		sum_execution_time = kernel_execution_time[rpts] + sum_execution_time;
	}

	//calculating the average execution time in milli seconds
	cl_float avg_execution_time = (cl_float)(sum_execution_time / (MAX_IERATIONS));
	printf("\nExecution done in GPU.\n");
	printf("Windows Performance Counter (WPC) API execution time.\n");
	printf("Average execution time %f ms.\n", avg_execution_time / 1000);

	//calculating standard deviation
	cl_ulong sum_deviation = 0;
	for (int val = 0; val < MAX_IERATIONS; val++) {
		//cl_ulong temp = ((kernel_execution_time[val] / 1000) - avg_execution_time);
		cl_long temp = (kernel_execution_time[val] - avg_execution_time);
		sum_deviation = sum_deviation + (temp * temp);

	}

	cl_float variance = (cl_float)(sum_deviation / (MAX_IERATIONS));
	cl_float standard_deviation = sqrt(variance);
	printf("Standard deviation time %f ms.\n", standard_deviation / 1000);

	err = clEnqueueReadImage(commands0, outputImg1, CL_TRUE, origin, region, 0, 0, hGreyScaleImage, 0, NULL, NULL);

	// NOTE: For stbi_write functions return value: 0 for ERROR, non-zero for SUCCESS
	outPath.append("greyscale_image.jpg");
	err = stbi_write_jpg(outPath.c_str(), imgCols, imgRows, imgChannels, hGreyScaleImage, 100);

	//serial implementation to generate a greyscale image
	unsigned char *greyscale_image_serial = (unsigned char*)malloc(imgRows*imgCols*imgChannels * sizeof(unsigned char));
	//generate noisy image from the grey scale image
	generateGreyscaleImage(imgdata, greyscale_image_serial, imgCols, imgRows, imgChannels, &sampler);

	//Gaussian Random distribution generator
	//mean = 0 and standard deviation = 2.0 * 2.0
	const float gaussianDistMean = 0;
	const float gaussianDistVariance = 2.0f;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<float> gaussianDistribution(gaussianDistMean, gaussianDistVariance);
	float *gaussian_random_distribution = (float *)malloc(imgRows* imgCols * sizeof(float));
	int val = 0;
	for (int val = 0; val < imgRows * imgCols; val++) {
		gaussian_random_distribution[val] = gaussianDistribution(generator);
	}


	//serial implementation to generate a noisy image
	unsigned char *noisy_image_serial = (unsigned char*)malloc(imgRows*imgCols*imgChannels * sizeof(unsigned char));
	//generate noisy image from the grey scale image
	generateNoisyImage(hGreyScaleImage, noisy_image_serial, imgCols, imgRows, imgChannels, &sampler, gaussian_random_distribution);

	//Run kernel2 to generate the denoised image from the noisy image
	err = clEnqueueWriteImage(commands0, inputImg3, CL_TRUE, origin, region, 0, 0, noisy_image_serial, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		printf("Error %d: clEnqueueWriteImage Failed!\n", err);
		clReleaseProgram(program3);
		clReleaseContext(context);
		return EXIT_FAILURE;
	}
	// Setting the arguments to our compute kernel in order to execute it. 
	err = clSetKernelArg(denoise_kernels[2], 0, sizeof(cl_mem), &inputImg3);
	//printf("Setting argument number 0\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 0!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[2], 1, sizeof(cl_mem), &outputImg3);
	//printf("Setting argument number 1\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 1!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[2], 2, sizeof(int), &imgCols);
	//printf("Setting argument number 2\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 2!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[2], 3, sizeof(int), &imgRows);
	//printf("Setting argument number 3\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 3!\n");
		return EXIT_FAILURE;
	}

	err = clSetKernelArg(denoise_kernels[2], 4, sizeof(cl_sampler), &sampler);
	//printf("Setting argument number 4\n");
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to set argument 4!\n");
		return EXIT_FAILURE;
	}

	//getting the profiling info
	ctr_start = 0, ctr_end = 0, freq = 0;
	sum_execution_time = 0;
	printf("\ndenoise_algorithm kernel performance evaluation.");
	for (int rpts = 0; rpts < MAX_IERATIONS; rpts++)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_start));
		err = clEnqueueNDRangeKernel(commands0, denoise_kernels[2], dim0, NULL, global0, local0, 0, NULL, NULL);

		if (CL_SUCCESS != err)
		{
			printf("Error %d: Failed to execute kernel!\n", err);
			clReleaseKernel(denoise_kernels[2]);
			clReleaseProgram(program3);
			clReleaseCommandQueue(commands0);
			clReleaseContext(context);
			return EXIT_FAILURE;
		}
		err = clFinish(commands0);
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_end));
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			clReleaseKernel(denoise_kernels[2]);
			clReleaseProgram(program3);
			clReleaseCommandQueue(commands0);
			clReleaseContext(context);
			return EXIT_FAILURE;
		}
		//To get more precise data we convert to micro s before dividing by ticks-per-second.
		cl_ulong temp = (ctr_end - ctr_start) * 1000000;
		kernel_execution_time[rpts] = (temp * (1.0 / freq));
		sum_execution_time = kernel_execution_time[rpts] + sum_execution_time;
	}

	//calculating the average execution time in milli seconds
	avg_execution_time = (cl_float)(sum_execution_time / (MAX_IERATIONS));
	printf("\nExecution done in GPU.\n");
	printf("Windows Performance Counter (WPC) API execution time.\n");
	printf("Average execution time %f ms.\n", avg_execution_time / 1000);

	//calculating standard deviation
	sum_deviation = 0;
	for (int val = 0; val < MAX_IERATIONS; val++) {
		//cl_ulong temp = ((kernel_execution_time[val] / 1000) - avg_execution_time);
		cl_long temp = (kernel_execution_time[val] - avg_execution_time);
		sum_deviation = sum_deviation + (temp * temp);

	}

	variance = (cl_float)(sum_deviation / (MAX_IERATIONS));
	standard_deviation = sqrt(variance);
	printf("Standard deviation time %f ms.\n", standard_deviation / 1000);
	//printf("\n\n***** NDRange is finished ***** \n");*/

	err = clEnqueueReadImage(commands0, outputImg3, CL_TRUE, origin, region, 0, 0, hDenoisedImage, 0, NULL, NULL);
	// NOTE: For stbi_write functions return value: 0 for ERROR, non-zero for SUCCESS
	outPath = "C:/EE524/project/VS/output/";
	outPath.append(outFilename);
	err = stbi_write_jpg(outPath.c_str(), imgCols, imgRows, imgChannels, hDenoisedImage, 100);


	//serial implementation in host side
	int num = imgCols * imgRows;
	// convert 1D imgdata to 2D
	unsigned char **noisy_image2D = (unsigned char**)malloc(3 * sizeof(unsigned char));
	for (int p = 0; p < 3; p++) {
		noisy_image2D[p] = &noisy_image_serial[p*num];
	}
	float stdDeviation = 15.0f, filterParameter = 4.0f;

	//denoise funtion serial implementaion
	denoise(noisy_image_serial, imgCols, imgRows, imgChannels, &sampler, stdDeviation, filterParameter);

	//Measure the effectgiveness of NLM algorithm using method noise
	//Method nise is the difference between the original image and the denoised output
	method_noise(hGreyScaleImage, hDenoisedImage, imgCols, imgRows, imgChannels, &sampler);

	clReleaseKernel(denoise_kernels[0]);
	clReleaseKernel(denoise_kernels[2]);
	clReleaseCommandQueue(commands0);

	clReleaseMemObject(inputImg1);
	clReleaseMemObject(inputImg3);
	clReleaseMemObject(outputImg1);
	clReleaseMemObject(outputImg3);

	clReleaseProgram(program1);
	clReleaseProgram(program3);
	clReleaseContext(context);

	//stbi_image_free(imgdata);
	free(kernel_source1);
	free(kernel_source3);
	free(hGreyScaleImage);
	free(hNoisyImage);
	stbi_image_free(imgdata);

	return 0;
}

cl_platform_id get_intel_platform(const string platformName)
{
	// Trying to get a handle to Intel's OpenCL platform using function 
	// Trying to identify one platform: 

	cl_platform_id platforms[10] = { NULL };
	cl_uint num_platforms = 0;

	cl_int err = clGetPlatformIDs(10, platforms, &num_platforms);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to get a platform id!\n");
		return NULL;
	}

	size_t returned_size = 0;
	cl_char platform_name[1024] = { 0 }, platform_prof[1024] = { 0 }, platform_vers[1024] = { 0 }, platform_exts[1024] = { 0 };

	for (unsigned int ui = 0; ui < num_platforms; ++ui)
	{
		// Found one platform. Query specific information about the found platform using the function  
		// Trying to query platform specific information... 

		err = clGetPlatformInfo(platforms[ui], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, &returned_size);
		err |= clGetPlatformInfo(platforms[ui], CL_PLATFORM_VERSION, sizeof(platform_vers), platform_vers, &returned_size);
		err |= clGetPlatformInfo(platforms[ui], CL_PLATFORM_PROFILE, sizeof(platform_prof), platform_prof, &returned_size);
		err |= clGetPlatformInfo(platforms[ui], CL_PLATFORM_EXTENSIONS, sizeof(platform_exts), platform_exts, &returned_size);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to get platform info!\n");
			return NULL;
		}

		// check for Intel platform 
		if (!strcmp((char*)platform_name, platformName.c_str())) {
			printf("\nPlatform information\n");
			printf(SEPARATOR);
			printf("Platform name:       %s\n", (char *)platform_name);
			printf("Platform version:    %s\n", (char *)platform_vers);
			printf("Platform profile:    %s\n", (char *)platform_prof);
			printf("Platform extensions: %s\n", ((char)platform_exts[0] != '\0') ? (char *)platform_exts : "NONE");
			return platforms[ui];
		}
	}

	return NULL;
}



void build_fail_log(cl_program program, cl_device_id device_id)
{
	cl_int err = CL_SUCCESS;
	size_t log_size = 0;

	err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	if (CL_SUCCESS != err)
	{
		printf("Error: Failed to read build log length...\n");
		return;
	}

	char* build_log = (char*)malloc(sizeof(char) * log_size + 1);
	if (NULL != build_log)
	{
		err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, &log_size);
		if (CL_SUCCESS != err)
		{
			printf("Error: Failed to read build log...\n");
			free(build_log);
			return;
		}

		build_log[log_size] = '\0';    // mark end of message string 

		printf("Build Log:\n");
		puts(build_log);
		fflush(stdout);

		free(build_log);
	}
}
void generateGreyscaleImage(unsigned char *input_img, unsigned char *greyscale_image, int cols, int rows, int channels, cl_sampler *sampler) {
	// This function adds Gaussian Noise samples with zero mean and a standard deviation of 1 to the input image
	//output is a noisy image
	string out_Path = "C:/EE524/project/VS/output/greyscale_image_serial_calculation.jpg";
	const float amplitude = 10.0f;
	printf("\n generate GreyscaleImage serial implementation performance evaluation.");
	//getting the profiling info
	cl_ulong ctr_start = 0, ctr_end = 0, freq = 0;
	cl_ulong sum_execution_time = 0;

	//calculating profiling info for 100 iterations
	cl_ulong kernel_execution_time[MAX_IERATIONS];
	//calculating profiling info for 500 iterations
	for (int rpts = 0; rpts < MAX_IERATIONS; rpts++)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_start));
		for (int rowNum = 0; rowNum < rows; rowNum++) {
			for (int colNum = 0; colNum < cols; colNum++) {
				float temp = 0;
				for (int channelNum = 0; channelNum < channels; channelNum++) {
					int index = (rowNum * cols + colNum)* channels + channelNum;
					//write resultant filtered pixel to output image
					unsigned char inp = input_img[index];
					if (channelNum == 1) {
						temp += 0.2126 * inp;
					}
					else if (channelNum == 2) {
						temp += 0.7152 * inp;
					}
					else if (channelNum == 3) {
						temp += 0.0722 * inp;
					}
				}
				greyscale_image[(rowNum * cols + colNum)* channels] = greyscale_image[(rowNum * cols + colNum)* channels + 1]
					= greyscale_image[(rowNum * cols + colNum)* channels + 2] = (unsigned char)temp;
				greyscale_image[(rowNum * cols + colNum)* channels + 3] = 1.0f;
			}
		}
		stbi_write_jpg(out_Path.c_str(), cols, rows, channels, greyscale_image, 100);
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_end));
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
		//To get more precise data we convert to micro s before dividing by ticks-per-second.
		cl_ulong temp = (ctr_end - ctr_start) * 1000000;
		kernel_execution_time[rpts] = (temp * (1.0 / freq));
		sum_execution_time = kernel_execution_time[rpts] + sum_execution_time;
	}

	//calculating the average execution time in milli seconds
	cl_float avg_execution_time = (cl_float)(sum_execution_time / (MAX_IERATIONS));
	printf("\nExecution done in serial.\n");
	printf("Windows Performance Counter (WPC) API execution time.\n");
	printf("Average execution time %f ms.\n", avg_execution_time / 1000);

	//calculating standard deviation
	cl_long sum_deviation = 0;
	for (int val = 0; val < MAX_IERATIONS; val++) {
		//cl_ulong temp = ((kernel_execution_time[val] / 1000) - avg_execution_time);
		cl_long temp = (kernel_execution_time[val] - avg_execution_time);
		sum_deviation = sum_deviation + (temp * temp);

	}

	cl_float variance = (cl_float)(sum_deviation / (MAX_IERATIONS));
	cl_float standard_deviation = sqrt(variance);
	printf("Standard deviation time %f ms.\n", standard_deviation / 1000);
}

void generateNoisyImage(unsigned char *input_Img, unsigned char *noisy_image, int cols, int rows, int channels, cl_sampler *sampler, float *gaussianRandomDist) {
	// This function adds Gaussian Noise samples with zero mean and a standard deviation of 1 to the input image
	//output is a noisy image
	string out_Path = "C:/EE524/project/VS/output/noisy_image_serial_calculation.jpg";
	const float amplitude = 10.0f;
	printf("\n generate noisy image serial implementation performance evaluation.");
	//getting the profiling info
	cl_ulong ctr_start = 0, ctr_end = 0, freq = 0;
	cl_ulong sum_execution_time = 0;

	//calculating profiling info for 100 iterations
	cl_ulong kernel_execution_time[MAX_IERATIONS];
	//calculating profiling info for 500 iterations
	for (int rpts = 0; rpts < MAX_IERATIONS; rpts++)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_start));
		for (int rowNum = 0; rowNum < rows; rowNum++) {
			for (int colNum = 0; colNum < cols; colNum++) {
				float temp = gaussianRandomDist[rowNum * cols + colNum];
				for (int channelNum = 0; channelNum < channels; channelNum++) {
					int index = (rowNum * cols + colNum)* channels + channelNum;
					//write resultant filtered pixel to output image
					if (channelNum + 1 != 4) {
						noisy_image[index] = input_Img[index] + amplitude * temp;
					}
					else {
						noisy_image[index] = 1.0f;
					}
				}
			}

		}
		stbi_write_jpg(out_Path.c_str(), cols, rows, channels, noisy_image, 100);
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_end));
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
		//To get more precise data we convert to micro s before dividing by ticks-per-second.
		cl_ulong temp = (ctr_end - ctr_start) * 1000000;
		kernel_execution_time[rpts] = (temp * (1.0 / freq));
		sum_execution_time = kernel_execution_time[rpts] + sum_execution_time;
	}

	//calculating the average execution time in milli seconds
	cl_float avg_execution_time = (cl_float)(sum_execution_time / (MAX_IERATIONS));
	printf("\nExecution done in serial.\n");
	printf("Windows Performance Counter (WPC) API execution time.\n");
	printf("Average execution time %f ms.\n", avg_execution_time / 1000);

	//calculating standard deviation
	cl_long sum_deviation = 0;
	for (int val = 0; val < MAX_IERATIONS; val++) {
		//cl_ulong temp = ((kernel_execution_time[val] / 1000) - avg_execution_time);
		cl_long temp = (kernel_execution_time[val] - avg_execution_time);
		sum_deviation = sum_deviation + (temp * temp);

	}

	cl_float variance = (cl_float)(sum_deviation / (MAX_IERATIONS));
	cl_float standard_deviation = sqrt(variance);
	printf("Standard deviation time %f ms.\n", standard_deviation / 1000);

}

void denoise(unsigned char *noisy_image, int cols, int rows, int channels,
	cl_sampler *sampler, float stdDeviation, float filterParameter)
{
	// convert 1D imgdata to 2D
	//removing A value from RGBA 
	unsigned char *noisy_image_modified = (unsigned char*)malloc(rows * cols * 3 * sizeof(unsigned char));
	for (int x = 0; x < rows; x++) {
		for (int y = 0; y < cols; y++) {
			int p = 0;
			for (int cNum = 0; cNum < channels; cNum++) {
				if (p < 3) {
					noisy_image_modified[(x* cols + y) * 3 + p] = noisy_image[(x* cols + y) * channels + cNum];
					p++;
				}
			}
		}
	}

	printf("\n generate denoised image serial implementation performance evaluation.");
	//getting the profiling info
	cl_ulong ctr_start = 0, ctr_end = 0, freq = 0;
	cl_ulong sum_execution_time = 0;

	//calculating profiling info for 100 iterations
	cl_ulong kernel_execution_time[MAX_IERATIONS];
	//calculating profiling info for 500 iterations
	for (int rpts = 0; rpts < MAX_IERATIONS; rpts++)
	{
		string out_Path = "C:/EE524/project/VS/output/denoised_image_serial_implementation.jpg";
		unsigned char *denoisedImg = (unsigned char*)malloc(rows * cols * channels * sizeof(unsigned char));
		int patchHalfWidth = (int)SIMILARITY_WINDOW / 2;
		int searchWindowHalfWidth = (int)SEARCH_WINDOW / 2;
		for (int x = 0; x < rows; x++) {
			for (int y = 0; y < cols; y++) {
				float normalizingconst = 0;
				float weights[SEARCH_WINDOW][SEARCH_WINDOW];
				for (int i = -searchWindowHalfWidth; i < searchWindowHalfWidth; i++) {
					for (int j = -searchWindowHalfWidth; j < searchWindowHalfWidth; j++) {
						if (i != x || j != y) {
							float euclideanDist = 0.0f;
							for (int cNum = 0; cNum < 3; cNum++) {
								float dist = 0.0;
								for (int p = -patchHalfWidth; p <= patchHalfWidth; p++) {
									for (int q = -patchHalfWidth; q <= patchHalfWidth; q++) {
										//calculating the Euclidean distance
										int tempX1 = x + p;
										int tempY1 = y + q;
										tempX1 = (tempX1 < 0) ? 0 : tempX1;
										tempY1 = (tempY1 < 0) ? 0 : tempY1;
										tempX1 = (tempX1 >= rows) ? rows - 1 : tempX1;
										tempY1 = (tempY1 >= cols) ? cols - 1 : tempY1;
										int tempX2 = i + p;
										int tempY2 = j + q;
										tempX2 = (tempX2 < 0) ? 0 : tempX2;
										tempY2 = (tempY2 < 0) ? 0 : tempY2;
										tempX2 = (tempX2 >= rows) ? rows - 1 : tempX2;
										tempY2 = (tempY2 >= cols) ? cols - 1 : tempY2;
										float diff = 0.0f;
										diff = noisy_image_modified[(tempX1* cols + tempY1) * 3 + cNum] - noisy_image_modified[(tempX2* cols + tempY2) * 3 + cNum];
										dist += diff * diff;
									}
								}
								euclideanDist = euclideanDist + dist;
							}
							//number of times the nonNormalizeddist was calculated
							int numExceution = 3 * SIMILARITY_WINDOW * SIMILARITY_WINDOW;
							//calculation of denoised signal Euclidean ditsnance
							float euclideanDistDenoised = euclideanDist - (2 * stdDeviation * stdDeviation * numExceution);
							//float euclideanDistDenoised = euclideanDist - (2 * stdDeviation * stdDeviation);
							if (euclideanDistDenoised < 0) {
								euclideanDistDenoised = 0;
							}
							//float nonNormalizedWeights = exp(-euclideanDistDenoised / (filterParameter* filterParameter));
							float nonNormalizedWeights = euclideanDistDenoised / (filterParameter* filterParameter * stdDeviation * stdDeviation*numExceution);
							//float nonNormalizedWeights = euclideanDistDenoised / (filterParameter* filterParameter);
							weights[i + searchWindowHalfWidth][j + searchWindowHalfWidth] = nonNormalizedWeights;
							normalizingconst += nonNormalizedWeights;
							//printf("weights %f  normalizingconst %f   nonNormalizedWeights %f \n", weights[i + searchWindowHalfWidth][j + searchWindowHalfWidth], normalizingconst, nonNormalizedWeights);
						}
					}
				}
				//printf("normalizingconst %f  \n", normalizingconst);
				//normalizing weights
				float sum;
				for (int i = -searchWindowHalfWidth; i < searchWindowHalfWidth; i++) {
					for (int j = -searchWindowHalfWidth; j < searchWindowHalfWidth; j++) {
						sum = 0.0f;
						float normalizedWeights = 0.0f;
						if (i != x || j != y) {
							for (int cNum = 0; cNum < 3; cNum++) {
								//intializing the weight with 0
							//weight is 0 if 2 patches are not at all similar and 1 if 2 patches are the same
								if (normalizingconst != 0.0f) {
									normalizedWeights = (weights[i + searchWindowHalfWidth][j + searchWindowHalfWidth] / normalizingconst);
								}
								//printf("normalizedWeights1 %f \t", normalizedWeights); 
								for (int p = -patchHalfWidth; p <= patchHalfWidth; p++) {
									for (int q = -patchHalfWidth; q <= patchHalfWidth; q++) {
										//calculating the Euclidean distance
										int tempX1 = x + p;
										int tempY1 = y + q;
										tempX1 = (tempX1 < 0) ? 0 : tempX1;
										tempY1 = (tempY1 < 0) ? 0 : tempY1;
										tempX1 = (tempX1 >= rows) ? rows - 1 : tempX1;
										tempY1 = (tempY1 >= cols) ? cols - 1 : tempY1;
										sum += normalizedWeights * noisy_image_modified[(tempX1 * cols + tempY1) * 3 + cNum];

									}
								}
							}
						}
					}
				}
				//printf(" %d\t", sum);
				for (int cNum = 0; cNum < channels; cNum++) {
					if (cNum + 1 != 4) {
						denoisedImg[(x*cols + y)* channels + cNum] = sum;
					}
					else {
						denoisedImg[(x*cols + y)* channels + cNum] = 1.0f;
					}
				}
			}
		}
		stbi_write_jpg(out_Path.c_str(), cols, rows, channels, denoisedImg, 100);
		QueryPerformanceCounter((LARGE_INTEGER *)(&ctr_end));
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
		//To get more precise data we convert to micro s before dividing by ticks-per-second.
		cl_ulong temp = (ctr_end - ctr_start) * 1000000;
		kernel_execution_time[rpts] = (temp * (1.0 / freq));
		sum_execution_time = kernel_execution_time[rpts] + sum_execution_time;
	}

	//calculating the average execution time in milli seconds
	cl_float avg_execution_time = (cl_float)(sum_execution_time / (MAX_IERATIONS));
	printf("\nExecution done in serial.\n");
	printf("Windows Performance Counter (WPC) API execution time.\n");
	printf("Average execution time %f ms.\n", avg_execution_time / 1000);

	//calculating standard deviation
	cl_long sum_deviation = 0;
	for (int val = 0; val < MAX_IERATIONS; val++) {
		//cl_ulong temp = ((kernel_execution_time[val] / 1000) - avg_execution_time);
		cl_long temp = (kernel_execution_time[val] - avg_execution_time);
		sum_deviation = sum_deviation + (temp * temp);

	}

	cl_float variance = (cl_float)(sum_deviation / (MAX_IERATIONS));
	cl_float standard_deviation = sqrt(variance);
	printf("Standard deviation time %f ms.\n", standard_deviation / 1000);

}

void method_noise(unsigned char *input_image, unsigned char *denoised_image, int cols, int rows, int channels,
	cl_sampler *sampler)
{
	// This function finds out the difference between the original image and the denoised output, that is the method noise
	//Method noise can be used as a measure of the effectiveness of the algorithm
	//output is a noisy image
	string out_Path = "C:/EE524/project/VS/output/method_noise_image.jpg";
	unsigned char *method_noise = (unsigned char*)malloc(rows * cols * channels * sizeof(unsigned char));
	for (int rowNum = 0; rowNum < rows; rowNum++) {
		for (int colNum = 0; colNum < cols; colNum++) {
			for (int channelNum = 0; channelNum < channels; channelNum++) {
				int index = (rowNum * cols + colNum)* channels + channelNum;
				//write resultant filtered pixel to output image
				if (channelNum + 1 != 4) {
					method_noise[index] = input_image[index] - denoised_image[index];
				}
				else {
					method_noise[index] = 1.0f;
				}
			}
		}

	}
	stbi_write_jpg(out_Path.c_str(), cols, rows, channels, method_noise, 100);
}