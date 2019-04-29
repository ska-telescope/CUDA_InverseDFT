
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
