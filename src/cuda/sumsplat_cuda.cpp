#include <torch/extension.h>

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

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor sumsplat_update_output(
		torch::Tensor input,
		torch::Tensor flow
) {
	CHECK_INPUT(input);
	CHECK_INPUT(flow);

	assert(flow.size(1) == 2);
	assert(input.size(2) == flow.size(2));
	assert(input.size(3) == flow.size(3));

	return sumsplat_update_output_cuda(input, flow);
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

	return sumsplat_update_gradinput_cuda(input, flow, grad_output);
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

	return sumsplat_update_gradflow_cuda(input, flow, grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sumsplat_update_output, "Summation splatting forward (CUDA)");
  m.def("backward_input", &sumspalt_update_gradinput, "Input gradients of summation splatting (CUDA)");
  m.def("backward_flow", &sumspalt_update_gradflow, "Flow gradients of summation splatting (CUDA)");
}
