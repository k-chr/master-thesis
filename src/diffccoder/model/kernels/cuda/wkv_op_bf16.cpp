#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void cuda_forward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y);
void cuda_backward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv);
void cuda_forward_with_state(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, float *s);
void cuda_state_forward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, float *last_state, bf16 *y, float *new_state);
void cuda_state_backward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, float *last_state, bf16 *gy, float *gnew_state, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv, float *glast_state);

void forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>());
}

void backward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
    torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(),
        gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
}

void forward_with_state(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    cuda_forward_with_state(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), s.data_ptr<float>());
}

void forward_state(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &last_state, torch::Tensor &y, torch::Tensor &new_state) {
    cuda_state_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), last_state.data_ptr<float>(), y.data_ptr<bf16>(), new_state.data_ptr<float>());
}

void backward_state(int64_t B, int64_t T, int64_t C,
                    torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &last_state,
                    torch::Tensor &gy, torch::Tensor &gnew_state, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &glast_state) {
    cuda_state_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), last_state.data_ptr<float>(),
        gy.data_ptr<bf16>(), gnew_state.data_ptr<float>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), glast_state.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
    m.def("backward", &backward, "wkv backward");
    m.def("forward_with_state", &forward_with_state, "wkv forward with state");
    m.def("forward_state", &forward_state, "wkv forward with state");
    m.def("backward_state", &backward_state, "wkv backward with state");
}

TORCH_LIBRARY(wkv, m) {
    m.def("forward", forward);
    m.def("forward_with_state", forward_with_state);
    m.def("backward", backward);
    m.def("forward_state", forward_state);
    m.def("backward_state", backward_state);
}
