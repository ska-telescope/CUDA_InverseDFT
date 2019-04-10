 
// Copyright 2019 Seth Hall, Adam Campbell, Andrew Ensor
// High Performance Computing Research Laboratory, 
// Auckland University of Technology (AUT)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef __cplusplus
extern "C" {
#endif

#ifndef INVERSE_DIRECT_FOURIER_TRANSFORM_H_
#define INVERSE_DIRECT_FOURIER_TRANSFORM_H_

#include <ctime>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

//define function for checking CUDA errors
#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

// Speed of light
#ifndef C
	#define C 299792458.0
#endif

//Define struct for the configuration
typedef struct Config {
	int vis_count;
	const char *output_image_file;
	const char *vis_file;
	bool psf_enabled;
	double image_size;
	int render_size;
	int x_render_offset;
	int y_render_offset;
	double cell_size;
	double uv_scale;
	double frequency_hz;
	bool enable_right_ascension;
	int vis_batch_size;
} Config;

//Define struct for visibility coordinate
typedef struct Visibility {
	double u;
	double v;
	double w;
} Visibility;

//used for grid(image) and visibility intensity
typedef struct Complex {
	double real;
	double imaginary;
} Complex;


void init_config(Config *config);

void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity);

void create_perfect_image(Config *config, Complex *grid, Visibility *visibilities, Complex *vis_intensity);

void save_grid_to_file(Config *config, Complex *grid);

__global__ void inverse_dft_with_w_correction(double2 *grid, size_t grid_pitch, const double3 *visibilities, const double2 *vis_intensity,
		int vis_count, int batch_count, int x_offset, int y_offset, int render_size, double cell_size);

__device__ double2 complex_multiply(double2 z1, double2 z2);

float time_difference_msec(struct timeval t0, struct timeval t1);

void unit_test_init_config(Config *config);

double unit_test_generate_approximate_image(void);

#endif /* INVERSE_DIRECT_FOURIER_TRANSFORM_H_ */

#ifdef __cplusplus
}
#endif

