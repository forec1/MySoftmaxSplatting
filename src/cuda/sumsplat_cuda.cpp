#include <torch/extension.h>

torch::Tensor sumsplat_update_output_cuda(
		torch::Tensor input,
		torch::Tensor flow
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sumsplat_update_output, "Summation splatting forward (CUDA)");
}
