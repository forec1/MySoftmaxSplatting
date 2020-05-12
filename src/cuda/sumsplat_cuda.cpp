#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

torch::Tensor sumsplat_update_output_cuda(
		torch::Tensor input,
		torch::Tensor flow
);

torch::Tensor sumsplat_update_gradinput_cuda(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
);

torch::Tensor sumsplat_update_gradflow_cuda(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
);

template <typename scalar_t>
torch::Tensor sumsplat_update_output_cpu(
		torch::Tensor input,
		torch::Tensor flow
) {
	torch::Tensor output = torch::zeros_like(input);
	int N_max = output.size(0);
	int C_max = output.size(1);
	int Y_max = output.size(2);
	int X_max = output.size(3);

	auto a_input = input.accessor<scalar_t, 4>();
	auto a_flow = flow.accessor<scalar_t, 4>();
	auto a_output = output.accessor<scalar_t, 4>();

	for(int N = 0; N < N_max; ++N) {
		for(int C = 0; C < C_max; ++C) {
			for(int qy = 0; qy < Y_max; ++qy) {
				for(int qx = 0; qx < X_max; ++qx) {
					scalar_t flt_output_x = (scalar_t) (qx) + a_flow[N][0][qy][qx];
					scalar_t flt_output_y = (scalar_t) (qy) + a_flow[N][1][qy][qx];

					int p_up_left_x = (int) floor(flt_output_x);
					int p_up_left_y = (int) floor(flt_output_y);

					int p_up_right_x = p_up_left_x + 1;
					int p_up_right_y = p_up_left_y;

					int p_down_left_x = p_up_left_x;
					int p_down_left_y = p_up_left_y + 1;

					int p_down_right_x = p_up_left_x + 1;
					int p_down_right_y = p_up_left_y + 1;

					scalar_t b_up_left = ((scalar_t) (p_down_right_x) - flt_output_x) * ((scalar_t) (p_down_right_y) - flt_output_y);
					scalar_t b_up_right = (flt_output_x - (scalar_t) (p_down_left_x)) * ((scalar_t) (p_down_left_y) - flt_output_y);
					scalar_t b_down_left = ((scalar_t) (p_up_right_x) - flt_output_x) * (flt_output_y - (scalar_t) (p_up_right_y));
					scalar_t b_down_right = (flt_output_x - (scalar_t) (p_up_left_x)) * (flt_output_y - (scalar_t) (p_up_left_y));

					if((p_up_left_x >= 0) & (p_up_left_x < output.size(3)) & (p_up_left_y >= 0) & (p_up_left_y < output.size(2))) {
						a_output[N][C][p_up_left_y][p_up_left_x] += a_input[N][C][qy][qx] * b_up_left;
					}
					if((p_up_right_x >= 0) & (p_up_right_x < output.size(3)) & (p_up_right_y >= 0) & (p_up_right_y < output.size(2))) {
						a_output[N][C][p_up_right_y][p_up_right_x] += a_input[N][C][qy][qx] * b_up_right;
					}
					if((p_down_left_x >= 0) & (p_down_left_x < output.size(3)) & (p_down_left_y >= 0) & (p_down_left_y < output.size(2))) {
						a_output[N][C][p_down_left_y][p_down_left_x] += a_input[N][C][qy][qx] * b_down_left;
					}
					if((p_down_right_x >= 0) & (p_down_right_x < output.size(3)) & (p_down_right_y >= 0) & (p_down_right_y < output.size(2))) {
						a_output[N][C][p_down_right_y][p_down_right_x] += a_input[N][C][qy][qx] * b_down_right;
					}
				}
			}
		}
	}
	return output;
}

template <typename scalar_t>
torch::Tensor sumsplat_update_gradinput_cpu(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
) {
	torch::Tensor grad_input = torch::zeros_like(input);
	int N_max = grad_input.size(0);
	int C_max = grad_input.size(1);
	int Y_max = grad_input.size(2);
	int X_max = grad_input.size(3);

	auto a_flow = flow.accessor<scalar_t, 4>();
	auto a_grad_input = grad_input.accessor<scalar_t, 4>();
	auto a_grad_output = grad_output.accessor<scalar_t, 4>();

	for(int N = 0; N < N_max; ++N) {
		for(int C = 0; C < C_max; ++C) {
			for(int qy = 0; qy < Y_max; ++qy) {
				for(int qx = 0; qx < X_max; ++qx) {
					scalar_t flt_output_x = (scalar_t) (qx) + a_flow[N][0][qy][qx];
					scalar_t flt_output_y = (scalar_t) (qy) + a_flow[N][1][qy][qx];

					int p_up_left_x = (int) floor(flt_output_x);
					int p_up_left_y = (int) floor(flt_output_y);

					int p_up_right_x = p_up_left_x + 1;
					int p_up_right_y = p_up_left_y;

					int p_down_left_x = p_up_left_x;
					int p_down_left_y = p_up_left_y + 1;

					int p_down_right_x = p_up_left_x + 1;
					int p_down_right_y = p_up_left_y + 1;

					scalar_t b_up_left = ((scalar_t) (p_down_right_x) - flt_output_x) * ((scalar_t) (p_down_right_y) - flt_output_y);
					scalar_t b_up_right = (flt_output_x - (scalar_t) (p_down_left_x)) * ((scalar_t) (p_down_left_y) - flt_output_y);
					scalar_t b_down_left = ((scalar_t) (p_up_right_x) - flt_output_x) * (flt_output_y - (scalar_t) (p_up_right_y));
					scalar_t b_down_right = (flt_output_x - (scalar_t) (p_up_left_x)) * (flt_output_y - (scalar_t) (p_up_left_y));

					scalar_t grad = 0.0;

					if((p_up_left_x >= 0) & (p_up_left_x < grad_input.size(3)) & (p_up_left_y >= 0) & (p_up_left_y < grad_input.size(2))) {
						grad += a_grad_output[N][C][p_up_left_y][p_up_left_x] * b_up_left;
					}
					if((p_up_right_x >= 0) & (p_up_right_x < grad_input.size(3)) & (p_up_right_y >= 0) & (p_up_right_y < grad_input.size(2))) {
						grad += a_grad_output[N][C][p_up_right_y][p_up_right_x] * b_up_right;
					}
					if((p_down_left_x >= 0) & (p_down_left_x < grad_input.size(3)) & (p_down_left_y >= 0) & (p_down_left_y < grad_input.size(2))) {
						grad += a_grad_output[N][C][p_down_left_y][p_down_left_x] * b_down_left;
					}
					if((p_down_right_x >= 0) & (p_down_right_x < grad_input.size(3)) & (p_down_right_y >= 0) & (p_down_right_y < grad_input.size(2))) {
						grad += a_grad_output[N][C][p_down_right_y][p_down_right_x] * b_down_right;
					}
					a_grad_input[N][C][qy][qx] = grad;
				}
			}
		}
	}
	return grad_input;
}

template <typename scalar_t>
torch::Tensor sumsplat_update_gradflow_cpu(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
) {
	torch::Tensor grad_flow = torch::zeros_like(flow);
	int N_max = grad_flow.size(0);
	int C_max = grad_flow.size(1);
	int Y_max = grad_flow.size(2);
	int X_max = grad_flow.size(3);

	auto a_input = input.accessor<scalar_t, 4>();
	auto a_flow = flow.accessor<scalar_t, 4>();
	auto a_grad_flow = grad_flow.accessor<scalar_t, 4>();
	auto a_grad_output = grad_output.accessor<scalar_t, 4>();

	for(int N = 0; N < N_max; ++N) {
		for(int C = 0; C < C_max; ++C) {
			for(int qy = 0; qy < Y_max; ++qy) {
				for(int qx = 0; qx < X_max; ++qx) {
					scalar_t flt_grad_flow = 0.0;

					scalar_t flt_output_x = (scalar_t) (qx) + a_flow[N][0][qy][qx];
					scalar_t flt_output_y = (scalar_t) (qy) + a_flow[N][1][qy][qx];

					int p_up_left_x = (int) (floor(flt_output_x));
					int p_up_left_y = (int) (floor(flt_output_y));

					int p_up_right_x = p_up_left_x + 1;
					int p_up_right_y = p_up_left_y;

					int p_down_left_x = p_up_left_x;
					int p_down_left_y = p_up_left_y + 1;

					int p_down_right_x = p_up_left_x + 1;
					int p_down_right_y = p_up_left_y + 1;

					scalar_t db_dF_up_left = 0.0;
					scalar_t db_dF_up_right = 0.0;
					scalar_t db_dF_down_left = 0.0;
					scalar_t db_dF_down_right = 0.0;

					if (C == 0) {
						db_dF_up_left = ((scalar_t) (-1.0)) * ((scalar_t) (p_down_right_y) - flt_output_y);
						db_dF_up_right = ((scalar_t) (+1.0)) * ((scalar_t) (p_down_left_y) - flt_output_y);
						db_dF_down_left = ((scalar_t) (-1.0)) * (flt_output_y - (scalar_t) (p_up_right_y));
						db_dF_down_right = ((scalar_t) (+1.0)) * (flt_output_y - (scalar_t) (p_up_left_y));
					} else if (C == 1) {
						db_dF_up_left = ((scalar_t) (p_down_right_x) - flt_output_x) * ((scalar_t) (-1.0));
						db_dF_up_right = (flt_output_x - (scalar_t) (p_down_left_x)) * ((scalar_t) (-1.0));
						db_dF_down_left = ((scalar_t) (p_up_right_x) - flt_output_x) * ((scalar_t) (+1.0));
						db_dF_down_right = (flt_output_x - (scalar_t) (p_up_left_x)) * ((scalar_t) (+1.0));
					}

					int C_max = grad_output.size(1);
					for (int Ci = 0; Ci < C_max; Ci += 1) {
						scalar_t flt_input = a_input[N][Ci][qy][qx];
						if ((p_up_left_x >= 0) & (p_up_left_x < grad_output.size(3)) & (p_up_left_y >= 0) & (p_up_left_y < grad_output.size(2))) {
							flt_grad_flow += flt_input * a_grad_output[N][Ci][p_up_left_y][p_up_left_x] * db_dF_up_left;
						}
						if ((p_up_right_x >= 0) & (p_up_right_x < grad_output.size(3)) & (p_up_right_y >= 0) & (p_up_right_y < grad_output.size(2))) {
							flt_grad_flow += flt_input * a_grad_output[N][Ci][p_up_right_y][p_up_right_x] * db_dF_up_right;
						}
						if ((p_down_left_x >= 0) & (p_down_left_x < grad_output.size(3)) & (p_down_left_y >= 0) & (p_down_left_y < grad_output.size(2))) {
							flt_grad_flow += flt_input * a_grad_output[N][Ci][p_down_left_y][p_down_left_x] * db_dF_down_left;
						}
						if ((p_down_right_x >= 0) & (p_down_right_x < grad_output.size(3)) & (p_down_right_y >= 0) & (p_down_right_y < grad_output.size(2))) {
							flt_grad_flow += flt_input * a_grad_output[N][Ci][p_down_right_y][p_down_right_x] * db_dF_down_right;
						}
					}
					a_grad_flow[N][C][qy][qx] = flt_grad_flow;
				}
			}
		}
	}
	return grad_flow;
}

torch::Tensor sumsplat_update_output(
		torch::Tensor input,
		torch::Tensor flow
) {
	CHECK_INPUT(input);
	CHECK_INPUT(flow);

	assert(flow.size(1) == 2);
	assert(input.size(2) == flow.size(2));
	assert(input.size(3) == flow.size(3));

	if(input.type().is_cuda()){
		return sumsplat_update_output_cuda(input, flow);
	}
	return sumsplat_update_output_cpu<double>(input, flow);
}

torch::Tensor sumspalt_update_gradinput(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
) {
	CHECK_INPUT(input);
	CHECK_INPUT(flow);
	CHECK_INPUT(grad_output);

	assert(flow.size(1) == 2);
	assert(input.size(2) == flow.size(2));
	assert(input.size(3) == flow.size(3));
	if(input.type().is_cuda()) {
		return sumsplat_update_gradinput_cuda(input, flow, grad_output);
	}
	return sumsplat_update_gradinput_cpu<double>(input, flow, grad_output);
}

torch::Tensor sumspalt_update_gradflow(
		torch::Tensor input,
		torch::Tensor flow,
		torch::Tensor grad_output
) {
	CHECK_INPUT(input);
	CHECK_INPUT(flow);
	CHECK_INPUT(grad_output);

	assert(flow.size(1) == 2);
	assert(input.size(2) == flow.size(2));
	assert(input.size(3) == flow.size(3));

	if(input.type().is_cuda()) {
		return sumsplat_update_gradflow_cuda(input, flow, grad_output);
	}
	return sumsplat_update_gradflow_cpu<double>(input, flow, grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sumsplat_update_output, "Summation splatting forward (CUDA)");
  m.def("backward_input", &sumspalt_update_gradinput, "Input gradients of summation splatting (CUDA)");
  m.def("backward_flow", &sumspalt_update_gradflow, "Flow gradients of summation splatting (CUDA)");
}
