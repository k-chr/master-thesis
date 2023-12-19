#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void cuda_forward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y);
void cuda_backward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv);
void cuda_forward_with_state(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, float *s);

void forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void backward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
    torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(),
        gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
}

void forward_with_state(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_forward_with_state(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), s.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
    m.def("backward", &backward, "wkv backward");
    m.def("forward_with_state", &forward_with_state, "wkv forward with state");
}

TORCH_LIBRARY(wkv, m) {
    m.def("forward", forward);
    m.def("forward_with_state", forward_with_state);
    m.def("backward", backward);
}
