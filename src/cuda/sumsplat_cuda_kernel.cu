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

// dIt[p]_dI0[q] = b(u)
template <typename scalar_t>
__global__ void sumsplat_update_gradinput_cuda_kernel(
		const int n,
		const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> flow,
		const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_output,
		torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_input
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < n; i += stride) {
		const int N = (i / grad_input.size(3) / grad_input.size(2) / grad_input.size(1)) % grad_input.size(0);
		const int C = (i / grad_input.size(3) / grad_input.size(2)) % grad_input.size(1);
		const int qy = (i / grad_input.size(3)) % grad_input.size(2);
		const int qx =  i % grad_input.size(3);

		float flt_output_x = (float) qx + flow[N][0][qy][qx];
		float flt_output_y = (float) qy + flow[N][1][qy][qx];

		int p_up_left_x = (int) floor(flt_output_x);
		int p_up_left_y = (int) floor(flt_output_y);

		int p_up_right_x = p_up_left_x + 1;
		int p_up_right_y = p_up_left_y;

		int p_down_left_x = p_up_left_x;
		int p_down_left_y = p_up_left_y + 1;

		int p_down_right_x = p_up_left_x + 1;
		int p_down_right_y = p_up_left_y + 1;

		float b_up_left = ((float) (p_down_right_x) - flt_output_x) * ((float) (p_down_right_y) - flt_output_y);
		float b_up_right = (flt_output_x - (float) (p_down_left_x)) * ((float) (p_down_left_y) - flt_output_y);
		float b_down_left = ((float) (p_up_right_x) - flt_output_x) * (flt_output_y - (float) (p_up_right_y));
		float b_down_right = (flt_output_x - (float) (p_up_left_x)) * (flt_output_y - (float) (p_up_left_y));

		float grad = 0.0;

		if((p_up_left_x >= 0) & (p_up_left_x < grad_input.size(3)) & (p_up_left_y >= 0) & (p_up_left_y < grad_input.size(2))) {
			grad += grad_output[N][C][p_up_left_y][p_up_left_x] * b_up_left;
		}
		if((p_up_right_x >= 0) & (p_up_right_x < grad_input.size(3)) & (p_up_right_y >= 0) & (p_up_right_y < grad_input.size(2))) {
			grad += grad_output[N][C][p_up_right_y][p_up_right_x] * b_up_right;
		}
		if((p_down_left_x >= 0) & (p_down_left_x < grad_input.size(3)) & (p_down_left_y >= 0) & (p_down_left_y < grad_input.size(2))) {
			grad += grad_output[N][C][p_down_left_y][p_down_left_x] * b_down_left;
		}
		if((p_down_right_x >= 0) & (p_down_right_x < grad_input.size(3)) & (p_down_right_y >= 0) & (p_down_right_y < grad_input.size(2))) {
			grad += grad_output[N][C][p_down_right_y][p_down_right_x] * b_down_right;
		}
		grad_input[N][C][qy][qx] = grad;
	}
}

//dIt[q]/dF0->t[q] = db(u)/dF0->t[q] * I0[q]
//db(u)/dF0->t[q] = max(0, 1 - |uy|) * {0 if |ux| >= 1, else -sgn(ux)}
template <typename scalar_t>
__global__ void sumsplat_update_gradflow_cuda_kernel(
		const int n,
		const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
		const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> flow,
		const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_output,
		torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_flow
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < n; i += stride) {
		const int N = (i / grad_flow.size(3) / grad_flow.size(2) / grad_flow.size(1)) % grad_flow.size(0);
		const int C = (i / grad_flow.size(3) / grad_flow.size(2)) % grad_flow.size(1);
		const int qy = (i / grad_flow.size(3)) % grad_flow.size(2);
		const int qx =  i % grad_flow.size(3);

		float flt_output_x = (float) qx + flow[N][0][qy][qx];
		float flt_output_y = (float) qy + flow[N][1][qy][qx];

		int p_up_left_x = (int) floor(flt_output_x);
		int p_up_left_y = (int) floor(flt_output_y);

		int p_up_right_x = p_up_left_x + 1;
		int p_up_right_y = p_up_left_y;

		int p_down_left_x = p_up_left_x;
		int p_down_left_y = p_up_left_y + 1;

		int p_down_right_x = p_up_left_x + 1;
		int p_down_right_y = p_up_left_y + 1;

		float db_dF_up_left = 0.0;
		float db_dF_up_right = 0.0;
		float db_dF_down_left = 0.0;
		float db_dF_down_right = 0.0;

		//TODO: MENI SE CINI DA JE GRESKA U PREDZNACIMA KAD USPOREDJIVAM S JEDNADBOM
		//grads with respect to x component of flow
		if(C == 0) {
			//px < (qx + F0->t[qx] => ux < 0 -> -1.0
			db_dF_up_left = ((float) (-1.0)) * ((float) (p_down_right_y) - flt_output_y);

			//px > (qx + F0->t[qx] => ux > 0 -> +1.0
			db_dF_up_right = ((float) (+1.0)) * ((float) (p_down_left_y) - flt_output_y);

			db_dF_down_left = ((float) (-1.0)) * (flt_output_y - (float) (p_up_right_y));
			db_dF_down_right = ((float) (+1.0)) * (flt_output_y - (float) (p_up_left_y));

			// grads with respect to y component of flow
		} else if (C == 1) {
			//py < (qx + F0->t[qy] => ux < 0 -> -1.0
			db_dF_up_left = ((float) (-1.0)) * ((float) (p_down_right_x) - flt_output_x);
			db_dF_up_right = ((float) (-1.0)) * (flt_output_x - (float) (p_down_left_x));

			//py > (qx + F0->t[qy] => ux < 0 -> -1.0
			db_dF_down_left = ((float) (+1.0)) * ((float) (p_up_right_x) - flt_output_x);
			db_dF_down_right = ((float) (+1.0)) * (flt_output_x - (float) (p_up_left_x));
		}
		float grad = 0.0;
		int C_size = grad_output.size(1);

		for(int Ci = 0; Ci < C_size; Ci += 1) {
			float flt_input = input[N][Ci][qy][qx];

			if((p_up_left_x >= 0) & (p_up_left_x < grad_output.size(3)) & (p_up_left_y >= 0) & (p_up_left_y < grad_output.size(2))) {
				grad += flt_input * grad_output[N][Ci][p_up_left_y][p_up_left_x] * db_dF_up_left;
			}
			if((p_up_right_x >= 0) & (p_up_right_x < grad_output.size(3)) & (p_up_right_y >= 0) & (p_up_right_y < grad_output.size(2))) {
				grad += flt_input * grad_output[N][Ci][p_up_right_y][p_up_right_x] * db_dF_up_right;
			}
			if((p_down_left_x >= 0) & (p_down_left_x < grad_output.size(3)) & (p_down_left_y >= 0) & (p_down_left_y < grad_output.size(2))) {
				grad += grad_output[N][Ci][p_down_left_y][p_down_left_x] * db_dF_down_left;
			}
			if((p_down_right_x >= 0) & (p_down_right_x < grad_output.size(3)) & (p_down_right_y >= 0) & (p_down_right_y < grad_output.size(2))) {
				grad += flt_input * grad_output[N][Ci][p_down_right_y][p_down_right_x] * db_dF_down_right;
			}
			grad_flow[N][C][qy][qx] = grad;
		}
	}
}


torch::Tensor sumsplat_update_output_cuda(
		torch::Tensor input,
		torch::Tensor flow
) {
	torch::Tensor output = torch::zeros_like(input);
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

torch::Tensor sumsplat_update_gradinput_cuda(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
){
	torch::Tensor grad_input = torch::zeros_like(input);
	const int N = grad_input.numel();

	const int blockSize = 256;
	const int numBlocks = 1024;

	sumsplat_update_gradinput_cuda_kernel<<<blockSize, numBlocks>>>(
		N,
		flow.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
		grad_output.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
		grad_input.packed_accessor32<float,4,torch::RestrictPtrTraits>()
	);

	cudaDeviceSynchronize();
	return grad_input;

}

torch::Tensor sumsplat_update_gradflow_cuda(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
){
	torch::Tensor grad_flow = torch::zeros_like(input);
	const int N = grad_flow.numel();

	const int blockSize = 256;
	const int numBlocks = 1024;

	sumsplat_update_gradflow_cuda_kernel<<<blockSize, numBlocks>>>(
		N,
		input.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
		flow.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
		grad_output.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
		grad_flow.packed_accessor32<float,4,torch::RestrictPtrTraits>()
	);

	cudaDeviceSynchronize();
	return grad_flow;

}
