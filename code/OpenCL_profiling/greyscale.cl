
__kernel void convert_to_greyscale(__read_only image2d_t inputImg, __write_only image2d_t outputImg, int cols, int rows, sampler_t sampler) {
	//this kernel converts a color image to greyscale
	//each execution modifies a single pixel in the image
	// use global IDs for output coords
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 pixel = read_imagef(inputImg, sampler, coords); // operate element-wise on all 3 color components (r,g,b) 
	float luminance = 0.2126*pixel.x + 0.7152*pixel.y + 0.0722*pixel.z;
	write_imagef(outputImg, coords, (float4)(luminance, luminance, luminance, 1.0f));
}
