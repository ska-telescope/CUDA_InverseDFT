
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "inverse_direct_fourier_transform.h"

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

/*
 * Intialize the configuration struct - IMPORTANT observation parameters
 */
void init_config(Config *config)
{	
	printf(">>> UPDATE: Loading configuration...\n\n");

	// Output files with paths for the resulting image, seperating real and imaginary components
	config->output_image_file      = "../output_image.csv";

	// Visibility Source file for the iDFT
	config->vis_file               = "../sample_visibilities.csv";

	// Flag to enable Point Spread Function
	config->psf_enabled            = false;

	// Frequency of visibility uvw terms in hertz
	config->frequency_hz           = 100e6;

	// specify single cell size in radians
	config->cell_size              = 6.39708380288950e-6;

	// the size (in pixels) of resulting image, where image coordinates range -image_size/2 to +image_size/2
	config->image_size             = 128.0;

	// if the image is too big, can specify a subregion where x_render and y_render offsets are the bottom corner of image
	// and render_size the amount of pixels from those coordinates in each direction you want your image. 0,0 is middle of image.
	config->render_size            = config->image_size;
	config->x_render_offset        = -(config->image_size / 2);
	config->y_render_offset        = -(config->image_size / 2);

	// Set the data to be right ascension, flips the u and w coordinate.
	config->enable_right_ascension = false;

	// set amount of visibilities per batch size, use 0 for no batching..
	// Will do a remainder batch if visibility count not divisible by batch size
	config->vis_batch_size         = 0;

	// Max number of threads per block dimension. Note this is 2D problem,
	// so actual number of full threads per block is this value squared.
	// i.e: 32^2 = 1024 threads per block - this is GPU specific.
	config->gpu_num_threads_per_block_dimension = 32;
}

/*
 * Function that performs iDFT on the GPU using CUDA, using visibilities (uvw) and their intensity
 * (real, imaginary) in the fourier domain to obtain sources in the image domain
 */
void create_perfect_image(Config *config, Complex *grid, Visibility *visibilities, Complex *vis_intensity)
{
	printf(">>> UPDATE:  Configuring iDFT and allocating GPU memory for grid (image)...\n\n");

	// Pointers for GPU memory
	double2 *d_g;
	double3 *v_k;
	double2 *vi_k;

	// use pitch to align data on GPU
	size_t g_pitch;
	CUDA_CHECK_RETURN(cudaMallocPitch(&d_g, &g_pitch, config->render_size * sizeof(Complex), config->render_size));

	// copy grid (image) to GPU using pitch
	CUDA_CHECK_RETURN((cudaMemcpy2D(d_g, g_pitch, grid, config->render_size * sizeof(Complex),
		config->render_size * sizeof(Complex), config->render_size, cudaMemcpyHostToDevice)));

	int max_threads_per_block_dim = min(config->gpu_num_threads_per_block_dimension, config->render_size);
	int num_blocks = (int) ceil((double) config->render_size / max_threads_per_block_dim);
	dim3 kernel_blocks(num_blocks, num_blocks, 1);
	dim3 kernel_threads(max_threads_per_block_dim, max_threads_per_block_dim, 1);

	// calculate number of batches and determine any remainder visibilities
	int number_of_batches = 1;
	int visibilities_on_last_batch = 0;
	int visibilities_per_batch = config->vis_count;
	if(config->vis_batch_size > 0)
	{ 	
		number_of_batches = config->vis_count/config->vis_batch_size;
		visibilities_on_last_batch = config->vis_count % config->vis_batch_size;
		visibilities_per_batch = config->vis_batch_size;
	}

	printf(">>> UPDATE: Setting %d batches of %d visibilities with a possible remainder batch of %d...\n\n",
			number_of_batches, visibilities_per_batch, visibilities_on_last_batch);
	
	// allocate CUDA memory for visibility data
	printf(">>> UPDATE: Allocating GPU memory for visibilities...\n\n");

	CUDA_CHECK_RETURN(cudaMalloc(&v_k, visibilities_per_batch * sizeof(Visibility)));

	CUDA_CHECK_RETURN(cudaMalloc(&vi_k,visibilities_per_batch * sizeof(Complex)));

	// Perform iDFT on each batch
	for(int i=0;i<number_of_batches;++i)
	{	
		int offset = i*	visibilities_per_batch;
		printf(">>> UPDATE: Batch %d : %d visibilities sent to GPU...\n\n",(i+1), visibilities_per_batch);

		CUDA_CHECK_RETURN(cudaMemcpy(v_k, visibilities+offset, visibilities_per_batch * sizeof(Visibility), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(vi_k, vis_intensity+offset, visibilities_per_batch  * sizeof(Complex), cudaMemcpyHostToDevice));

		cudaDeviceSynchronize();

		printf(">>> UPDATE: Executing iDFT CUDA kernel...\n\n");

		// iDFT performed on a portion of the image (assuming 0,0 in center of image), coordinates are from x and y render offset
		// origin to render size - Image must be square
		inverse_dft_with_w_correction<<<kernel_blocks, kernel_threads>>>(d_g, g_pitch, v_k, vi_k,
			config->vis_count, visibilities_per_batch, config->x_render_offset,
			config->y_render_offset, config->render_size, config->cell_size);
		cudaDeviceSynchronize();

		printf(">>> UPDATE: Completed %d batch \n\n",(i+1));
	}

	// Perform iDFT on any remainder visibility data
	if(visibilities_on_last_batch > 0)
	{
		printf(">>> UPDATE: Sending %d remaining visibilities for final processing...  \n\n",  visibilities_on_last_batch);
		int offset = number_of_batches * visibilities_per_batch;
		printf(">>> UPDATE: final batch of %d visibilities sent to GPU...\n\n", visibilities_on_last_batch);

		CUDA_CHECK_RETURN(cudaMemcpy(v_k, (Visibility*)visibilities + offset,
			visibilities_on_last_batch * sizeof(Visibility), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(vi_k, (Complex*)vis_intensity + offset,
			visibilities_on_last_batch  * sizeof(Complex), cudaMemcpyHostToDevice));

		cudaDeviceSynchronize();

		printf(">>> UPDATE: Executing final iDFT CUDA kernel call...\n\n");
		inverse_dft_with_w_correction<<<kernel_blocks, kernel_threads>>>(d_g, g_pitch, v_k, vi_k,
			config->vis_count, visibilities_on_last_batch, config->x_render_offset,
			config->y_render_offset, config->render_size, config->cell_size);
		cudaDeviceSynchronize();
	}

	// copy image data back from the GPU to CPU
	printf(">>> UPDATE: Batch processing complete...  \n\n");
	printf(">>> UPDATE: Copying image data from GPU back to host...\n\n");
	CUDA_CHECK_RETURN(cudaMemcpy2D(grid, config->render_size * sizeof(double2), d_g, g_pitch,
		config->render_size * sizeof(double2),
		config->render_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	printf(">>> UPDATE: Cleaning up allocated GPU memory...\n\n");
	CUDA_CHECK_RETURN(cudaFree(d_g));
	CUDA_CHECK_RETURN(cudaFree(v_k));
	CUDA_CHECK_RETURN(cudaFree(vi_k));
}

// Populates visibility array from file
void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity)
{
	printf(">>> UPDATE: Loading Visibilities from File: %s ...\n\n", config->vis_file);
	FILE *file = fopen(config->vis_file, "r");

	if(file == NULL)
	{
		printf(">>> ERROR: Unable to load sources from file...\n\n");
		return;
	}
	else
	{	// read the amount of visibilities in file
		fscanf(file, "%d\n", &(config->vis_count));

		// allocate visibility u,v,w and associated intensity value
		*visibilities = (Visibility*) calloc(config->vis_count, sizeof(Visibility));
		*vis_intensity = (Complex*) calloc(config->vis_count, sizeof(Complex));
		if(*visibilities == NULL || *vis_intensity == NULL) 
		{	
			fclose(file);
			return;
		}

		double wavelength_factor = config->frequency_hz / C;
		double temp_uu     = 0.0;
		double temp_vv     = 0.0; 
		double temp_ww     = 0.0;
		double temp_real   = 0.0;
		double temp_imag   = 0.0; 
		double temp_weight = 0.0;

		double right_asc_factor = (config->enable_right_ascension) ? -1.0 : 1.0;

		for(int i = 0; i < config->vis_count; i++)
		{
			fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &temp_uu, &temp_vv, &temp_ww,
				&temp_real, &temp_imag, &temp_weight);

			// Flip u and v coordinate if data should be right ascension
			temp_uu = temp_uu * right_asc_factor;
			temp_ww = temp_ww * right_asc_factor;

			// use weight to adjust the real, imag or set real to one if psf enabled
			if(config->psf_enabled)
			{
				temp_real = 1.0;
				temp_imag = 0.0;
			}
			else
			{	temp_real = temp_real * temp_weight;
				temp_imag = temp_imag * temp_weight;
			}
			// convert uvw from meters to wavelengths
			temp_uu = temp_uu * wavelength_factor;
			temp_vv = temp_vv * wavelength_factor;
			temp_ww = temp_ww * wavelength_factor;

			(*visibilities)[i] = (Visibility){.u = temp_uu, .v = temp_vv, .w = temp_ww};
			(*vis_intensity)[i] = (Complex){.real = temp_real, .imaginary = temp_imag};
		}

		fclose(file);
		printf(">>> UPDATE: Successfully read %d visibilities from file...\n\n",config->vis_count);
	}

}


// saves a csv file of the rendered iDFT region only
void save_grid_to_file(Config *config, Complex *grid)
{
    FILE *file_real = fopen(config->output_image_file, "w");
    double real_sum = 0.0;
    double imag_sum = 0.0;

    if(!file_real)
	{	
		printf(">>> ERROR: Unable to create grid file %s, check file structure exists...\n\n",
			config->output_image_file);
		return;
	}

    for(unsigned int r= 0; r < config->render_size; r++)
    {
    	for(unsigned int c = 0; c < config->render_size; c++)
        {
            Complex grid_point = grid[r*config->render_size+c];

            if(c<(config->render_size-1))
            	fprintf(file_real, "%.15f,", grid_point.real);
            else
            	fprintf(file_real, "%.15f", grid_point.real);

            real_sum += grid_point.real;
            imag_sum += grid_point.imaginary;
        }

        fprintf(file_real, "\n");
    }
    fclose(file_real);

    printf(">>> UPDATE: RealSum: %f, ImagSum: %f...\n\n", real_sum, imag_sum);
}

// visibilities (x,y,z) = (uu,vv,ww), visIntensity (x,y) = (real, imaginary), grid (x,y) = (real, imaginary)
__global__ void inverse_dft_with_w_correction(double2 *grid, size_t grid_pitch, const double3 *visibilities,
		const double2 *vis_intensity, int vis_count, int batch_count, int x_offset, int y_offset,
		int render_size, double cell_size)
{
	// look up id of thread
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	if(idx >= render_size || idy >= render_size)
		return;

	double real_sum = 0;
	double imag_sum = 0;

	// convert to x and y image coordinates
	double x = (idx+x_offset) * cell_size;
    double y = (idy+y_offset) * cell_size;

	double2 vis;
	double2 theta_complex = make_double2(0.0, 0.0);

	// precalculate image correction and wCorrection
	double image_correction = sqrt(1.0 - (x * x) - (y * y));
	double w_correction = image_correction - 1.0;

	// NOTE: below is an approximation... Uncomment if needed
	// double wCorrection = -((x*x)+(y*y))/2.0;

	// loop through all visibilities and create sum using iDFT formula
	for(int i = 0; i < batch_count; ++i)
	{	
		double theta = 2.0 * M_PI * (x * visibilities[i].x + y * visibilities[i].y
				+ (w_correction * visibilities[i].z));
		sincos(theta, &(theta_complex.y), &(theta_complex.x));
		vis = complex_multiply(vis_intensity[i], theta_complex);
		real_sum += vis.x;
		imag_sum += vis.y;
	}

	// adjust sum by image correction
	real_sum *= image_correction;
	imag_sum *= image_correction;

	// look up destination in image (grid) and divide by amount of visibilities (N)
	double2 *row = (double2*)((char*)grid + idy * grid_pitch);
	row[idx].x += (real_sum / vis_count);
	row[idx].y += (imag_sum / vis_count);
}

// done on GPU, performs a complex multiply of two complex numbers
__device__ double2 complex_multiply(double2 z1, double2 z2)
{
    double real = z1.x*z2.x - z1.y*z2.y;
    double imag = z1.y*z2.x + z1.x*z2.y;
    return make_double2(real, imag);
}

// used for performance testing to return the difference in milliseconds between two timeval structs
float time_difference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}


//**************************************//
//      UNIT TESTING FUNCTIONALITY      //
//**************************************//

void unit_test_init_config(Config *config)
{	
	config->output_image_file      = "../unit_test_500_sources_1024_grid_real.csv";
	config->vis_file               = "../unit_test_visibilities_500_sources.txt";
	config->psf_enabled            = false;
	config->frequency_hz           = 300e06;
	config->cell_size              = 4.848136811095360e-06;
	config->image_size             = 1024.0;
	config->render_size            = 1024;
	config->x_render_offset        = -512;
	config->y_render_offset        = -512;
	config->enable_right_ascension = false;
	config->vis_batch_size         = 0;
	config->gpu_num_threads_per_block_dimension = 32;
}

double unit_test_generate_approximate_image()
{
	// used to invalidate unit test
	double error = DBL_MAX;

	// Init config
	Config config;
	unit_test_init_config(&config);

	// Allocate grid mem
	Complex *grid = (Complex*) calloc(config.render_size * config.render_size, sizeof(Complex));
	if(grid == NULL)
	{	
		printf(">>> ERROR: Unable to allocate grid memory \n");
		return error;
	}

	// Read visibilities into memory
	Visibility *visibilities = NULL;
	Complex *vis_intensity = NULL;
	load_visibilities(&config, &visibilities, &vis_intensity);

	if(visibilities == NULL || vis_intensity == NULL)
	{	
		printf(">>> ERROR: Visibility memory was unable to be allocated \n\n");
		if(visibilities)      free(visibilities);
		if(vis_intensity)      free(vis_intensity);
		if(grid)			  free(grid);
		return error;
	}

	// Begin primitive timing...
	 struct timeval begin;
	gettimeofday(&begin,0);

	//Pefrom iDFT to obtain sources from visibility data
	create_perfect_image(&config, grid, visibilities, vis_intensity);
	
	// End primitive timing...
	 struct timeval end;
	 gettimeofday(&end,0);
	float time_diff = time_difference_msec(begin,end);
	 printf(">>> UPDATE: iDFT complete, time taken %f milliseconds or %f seconds\n\n",time_diff, time_diff/1000);

	// Compare image with "output image"
	FILE *file = fopen(config.output_image_file, "r");
	if(file == NULL)
	{
		printf(">>> ERROR: Unable to open unit test file %s...\n\n", config.output_image_file);
	    if(visibilities)      free(visibilities);
		if(vis_intensity)     free(vis_intensity);
		if(grid)			  free(grid);
		return error;
	}

	// Perform analysis
	double difference = 0.0;
	double file_image_element = 0.0;
	double bottom = 0.0;
	for(int row_indx = 0; row_indx < config.render_size; ++row_indx)
	{
		for(int col_indx = 0; col_indx < config.render_size; ++col_indx)
		{
			int lookup_indx = row_indx * config.render_size + col_indx;

			if(col_indx < config.render_size-1)
				fscanf(file, "%lf,", &file_image_element);
			else
				fscanf(file, "%lf", &file_image_element);
			
			difference += pow(file_image_element - grid[lookup_indx].real,2.0);
			bottom += pow(file_image_element,2.0);
		}
		fscanf(file, "\n");
	}

	difference = sqrt(difference) / sqrt(bottom);
	fclose(file);

	// clean up
    if(visibilities)      free(visibilities);
	if(vis_intensity)     free(vis_intensity);
	if(grid)			  free(grid);

	printf(">>> INFO: Measured absolute difference across full image: %lf...\n\n", difference);
	return difference;
}
