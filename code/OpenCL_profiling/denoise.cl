
__constant int SIMILARITY_WINDOW_RADIUS =  3;
__constant int SIMILARITY_WINDOW = 7U;
__constant int SEARCH_WINDOW = 21U;
__constant int SEARCH_WINDOW_RADIUS = 10U;
__constant float FILTER_PARAMETER = .4f;


__kernel void denoise_algorithm(__read_only image2d_t inputImg, __write_only image2d_t outputImg, int cols, int rows, sampler_t sampler)
{
	//this kernel converts a noisy image to a denoised image
	//each execution modifies a single pixel in the image

	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if(x < cols && y < rows){
		float weight, euclidean_distance = 0;
		for (int i = -SEARCH_WINDOW_RADIUS; i < SEARCH_WINDOW_RADIUS; i++) {
			for (int j = -SEARCH_WINDOW_RADIUS; j < SEARCH_WINDOW_RADIUS; j++) {
			//looping within the search window
				if (i != x || j != y) {
					float euclideanDist = 0.0f;
					float dist = 0.0;
					for (int p = -SIMILARITY_WINDOW_RADIUS; p <= SIMILARITY_WINDOW_RADIUS; p++) {
						for (int q = -SIMILARITY_WINDOW_RADIUS; q <= SIMILARITY_WINDOW_RADIUS; q++) {
						//looping within each patch/ similarity window
							float3 neighbor_pixel, pixel;
							if((x+ p < rows) && (y + q < cols))
								pixel = read_imagef(inputImg, sampler, (int2)(x+j, y+i)).xyz;
							if((i + p < rows) && (j + q < cols))
								neighbor_pixel = read_imagef(inputImg, sampler, (int2)(i + p, j + q)).xyz;
							//comparing patches to find the Euclidean distance
							euclidean_distance += (pixel.x -neighbor_pixel.x) * (pixel.x -neighbor_pixel.x) + (pixel.y -neighbor_pixel.y) * (pixel.y -neighbor_pixel.y)
												+ (pixel.z -neighbor_pixel.z) * (pixel.z -neighbor_pixel.z);
						}
					}
					//finding the weight for each patch from the Euclidean distance
					weight = native_exp(-(euclidean_distance /(FILTER_PARAMETER * FILTER_PARAMETER)));
					// normalizing constant value is the sum of all the weights in a patch
					float normalizing_constant;
					float3 denoised_pixel = {0,0,0};
					int p = 0;
					for (float i = -SIMILARITY_WINDOW_RADIUS; i <= SIMILARITY_WINDOW_RADIUS + 1; i++){
						for (float j = -SIMILARITY_WINDOW_RADIUS; j <= SIMILARITY_WINDOW_RADIUS + 1; j++){
							//multiplying all pixel values with the calculated weight
							p++;
	    					float3 pixel = read_imagef(inputImg, sampler, (int2)(x + j , y + i)).xyz;
							//value of each pixel is the sum of neighboring pixel values multiplied by their weights normalized to the total weight
	    					denoised_pixel.x += pixel.x * weight;
	    					denoised_pixel.y += pixel.y * weight;
	    					denoised_pixel.z += pixel.z * weight;
							normalizing_constant += weight;
	    				}
					}
					//dividing by normalizing constants to get a normalized value
					normalizing_constant = 1.0f / normalizing_constant;
					denoised_pixel.x *= normalizing_constant;
					denoised_pixel.y *= normalizing_constant;
					denoised_pixel.z *= normalizing_constant;

					write_imagef(outputImg, (int2)(x, y), (float4)(denoised_pixel.x , denoised_pixel.y, denoised_pixel.z, 1.0f));
				}
			}
		}
	}
}