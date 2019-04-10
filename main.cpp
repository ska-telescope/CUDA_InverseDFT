
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

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <sys/time.h>

#include "inverse_direct_fourier_transform.h"

int main(int argc, char **argv)
{
    printf("=============================================================================\n");
	printf(">>> AUT HPC Research Laboratory - Inverse Direct Fourier Transform (CUDA) <<<\n");
	printf("=============================================================================\n\n");

	//initialize config struct
	Config config;
	init_config(&config);

	//allocate the grid or image of size render_size squared.. Each element is a Complex type
	printf(">>> UPDATE: Allocating Image (grid), size = %d by %d as flat memory..\n\n",
			config.render_size,config.render_size);
	Complex *grid = (Complex*) calloc(config.render_size*config.render_size,sizeof(Complex));

	if(grid == NULL)
	{	
		printf(">>> ERROR: Unable to allocate grid memory \n");
		return EXIT_FAILURE;
	}

	//Try to load visibilities from file and their intensities
	Visibility *visibilities = NULL;
	Complex *vis_intensity = NULL;
	load_visibilities(&config, &visibilities, &vis_intensity);

	if(visibilities == NULL || vis_intensity == NULL)
	{	
		printf(">>> ERROR: Visibility memory was unable to be allocated \n\n");
		if(visibilities)      free(visibilities);
		if(vis_intensity)      free(vis_intensity);
		if(grid)			  free(grid);
		return EXIT_FAILURE;
	}

	struct timeval begin;
	gettimeofday(&begin,0);

	// Perform inverse direct fourier transform to 
	// obtain sources from visibilities (as image)
	create_perfect_image(&config, grid, visibilities, vis_intensity);

	struct timeval end;
	gettimeofday(&end,0);
	float time_diff = time_difference_msec(begin,end);
	printf(">>> UPDATE: Inverse DFT complete, time taken: %f milliseconds or %f seconds...\n\n",time_diff, time_diff/1000);
	printf(">>> UPDATE: Saving rendered portion of image to file... %s\n\n", config.output_image_file);
    save_grid_to_file(&config, grid);

    // Clean up
    if(visibilities)      free(visibilities);
	if(vis_intensity)      free(vis_intensity);
	if(grid)			  free(grid);

    printf(">>> UPDATE: Inverse Direct Fourier Transform operations complete, exiting...\n\n");
	return EXIT_SUCCESS;
}
