#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void sumsplat_update_output_cuda_kernel(
		const int n,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> flow,
		torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < n; i += stride) {
		const int N = (i / output.size(3) / output.size(2) / output.size(1)) % output.size(0);
		const int C = (i / output.size(3) / output.size(2)) % output.size(1);
		const int qy = (i / output.size(3)) % output.size(2);
		const int qx =  i % output.size(3);

		/* Float vrijednosti indeksa piksela:
		 * q + F_0->t[q]
		 * */
		float flt_output_x = (float) qx + flow[N][0][qy][qx];
		float flt_output_y = (float) qy + flow[N][1][qy][qx];

		// Indeksi susjednih piksela
		int p_up_left_x = (int) floor(flt_output_x);
		int p_up_left_y = (int) floor(flt_output_y);

		int p_up_right_x = p_up_left_x + 1;
		int p_up_right_y = p_up_left_y;

		int p_down_left_x = p_up_left_x;
		int p_down_left_y = p_up_left_y + 1;

		int p_down_right_x = p_up_left_x + 1;
		int p_down_right_y = p_up_left_y + 1;

		/*
		 * Doprinos za svaki od susjednih piksela. Za racunanje doprinosa za trenutni piksel
		 * uzima se udaljenost izmedu nasuprotnog piksela i float vrijednosti indeksa piksela
		 * (flt_output_x, flt_output_y).
		 * Uzima se nasuprotni piksel jer je doprinos izražen kao (1 - |ux|)*(1 - |uy|), tj.
		 * ako je float vrijednost indeksa piksela bliža trenutnom pikselu on treba doprinosit više,
		 * zato se radi |nasuprotni_piksel - float_indeks_piksela|
		 * */
		float b_up_left = ((float) (p_down_right_x) - flt_output_x) * ((float) (p_down_right_y) - flt_output_y);
		float b_up_right = (flt_output_x - (float) (p_down_left_x)) * ((float) (p_down_left_y) - flt_output_y);
		float b_down_left = ((float) (p_up_right_x) - flt_output_x) * (flt_output_y - (float) (p_up_right_y));
		float b_down_right = (flt_output_x - (float) (p_up_left_x)) * (flt_output_y - (float) (p_up_left_y));

		if((p_up_left_x >= 0) & (p_up_left_x < output.size(3)) & (p_up_left_y >= 0) & (p_up_left_y < output.size(2))) {
			atomicAdd(&output[N][C][p_up_left_y][p_up_left_x],( input[N][C][qy][qx] * b_up_left));
		}
		if((p_up_right_x >= 0) & (p_up_right_x < output.size(3)) & (p_up_right_y >= 0) & (p_up_right_y < output.size(2))) {
			atomicAdd(&output[N][C][p_up_right_y][p_up_right_x], (input[N][C][qy][qx] * b_up_right));
		}
		if((p_down_left_x >= 0) & (p_down_left_x < output.size(3)) & (p_down_left_y >= 0) & (p_down_left_y < output.size(2))) {
			atomicAdd(&output[N][C][p_down_left_y][p_down_left_x],(input[N][C][qy][qx] * b_down_left));
		}
		if((p_down_right_x >= 0) & (p_down_right_x < output.size(3)) & (p_down_right_y >= 0) & (p_down_right_y < output.size(2))) {
			atomicAdd(&output[N][C][p_down_right_y][p_down_right_x], (input[N][C][qy][qx] * b_down_right));
		}
	}

}


torch::Tensor sumsplat_update_output_cuda(
		torch::Tensor input,
		torch::Tensor flow
) {
	torch::Tensor output = input.new_zeros(input.sizes());
	const int N = output.numel();

	const int blockSize = 256;
	const int numBlocks = 1024;

	// RuntimeError: CUDA error: invalid configuration argument
	// Pretpostavljam da nemam dovoljno jak GPU, zato sam stavio
	// numBlocks = 1024, a ne:
//	const int numBlocks = (N + blockSize - 1) / blockSize;

//	AT_DISPATCH_FLOATING_TYPES(input.type(), "zbroji", ([&] {
//		sumsplat_update_output_cuda_kernel<scalar_t><<<blockSize, numBlocks>>>(
//				N,
//				input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//				flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//				output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
//		);
//	}));

	sumsplat_update_output_cuda_kernel<float><<<blockSize, numBlocks>>>(
			N,
			input.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
			flow.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
			output.packed_accessor32<float,4,torch::RestrictPtrTraits>()
	);


	cudaDeviceSynchronize();

	return output;
}
