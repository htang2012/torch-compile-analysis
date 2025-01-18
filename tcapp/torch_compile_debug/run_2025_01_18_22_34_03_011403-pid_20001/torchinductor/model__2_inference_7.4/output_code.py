# AOT ID: ['2_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


cpp_fused_convolution_relu_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'const int64_t'], '''
#include "/tmp/torchinductor_root/tmptex2y7qd/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       const int64_t ks0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(8L))
                {
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(672L); x2+=static_cast<int64_t>(8L))
                    {
                        alignas(8) float tmp2[8*8];
                        for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (676L*x1) + (676L*x1_inner) + (21632L*x0)), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            tmp1.store(tmp2 + static_cast<int64_t>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp2, static_cast<int64_t>(8), out_ptr0 + static_cast<int64_t>(x1 + (32L*x2) + (21632L*x0)), static_cast<int64_t>(32L));
                    }
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(672L); x2<static_cast<int64_t>(676L); x2+=static_cast<int64_t>(4L))
                    {
                        alignas(8) float tmp2[8*8];
                        for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (676L*x1) + (676L*x1_inner) + (21632L*x0)), static_cast<int64_t>(4L));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            tmp1.store(tmp2 + static_cast<int64_t>(4L*x1_inner), static_cast<int64_t>(4L));
                        }
                        at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(4L)>(tmp2, static_cast<int64_t>(4L), out_ptr0 + static_cast<int64_t>(x1 + (32L*x2) + (21632L*x0)), static_cast<int64_t>(32L));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(8L))
                {
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(8L); x2+=static_cast<int64_t>(8L))
                    {
                        alignas(8) float tmp1[8*8];
                        for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)), static_cast<int64_t>(8));
                            tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp1, static_cast<int64_t>(8), out_ptr1 + static_cast<int64_t>(x1 + (32L*x2) + (288L*x0)), static_cast<int64_t>(32L));
                    }
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(8L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(1L))
                    {
                        alignas(8) float tmp1[8*8];
                        for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)), static_cast<int64_t>(1L));
                            tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                        }
                        at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L)>(tmp1, static_cast<int64_t>(1L), out_ptr1 + static_cast<int64_t>(x1 + (32L*x2) + (288L*x0)), static_cast<int64_t>(32L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_1 = async_compile.cpp_pybinding(['const float*', 'float*', 'const int64_t'], '''
#include "/tmp/torchinductor_root/tmptex2y7qd/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       const int64_t ks0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(12L); x1+=static_cast<int64_t>(1L))
                {
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(8L); x2+=static_cast<int64_t>(8L))
                    {
                        #pragma GCC ivdep
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(64L); x3+=static_cast<int64_t>(8L))
                        {
                            alignas(8) float tmp11[8*8];
                            for (long x2_inner = 0; x2_inner < static_cast<int64_t>(8); x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(64L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(1536L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(1600L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                                auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                                auto tmp4 = at::vec::maximum(tmp3, tmp1);
                                auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                                auto tmp7 = at::vec::maximum(tmp6, tmp4);
                                auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                                auto tmp10 = at::vec::maximum(tmp9, tmp7);
                                tmp10.store(tmp11 + static_cast<int64_t>(8L*x2_inner));
                            }
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp11, static_cast<int64_t>(8), out_ptr0 + static_cast<int64_t>(x2 + (12L*x1) + (144L*x3) + (9216L*x0)), static_cast<int64_t>(144L));
                        }
                    }
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(8L); x2<static_cast<int64_t>(12L); x2+=static_cast<int64_t>(4L))
                    {
                        #pragma GCC ivdep
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(64L); x3+=static_cast<int64_t>(8L))
                        {
                            alignas(8) float tmp11[8*8];
                            for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4L); x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(64L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(1536L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(1600L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                                auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                                auto tmp4 = at::vec::maximum(tmp3, tmp1);
                                auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                                auto tmp7 = at::vec::maximum(tmp6, tmp4);
                                auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                                auto tmp10 = at::vec::maximum(tmp9, tmp7);
                                tmp10.store(tmp11 + static_cast<int64_t>(8L*x2_inner));
                            }
                            at::vec::transpose_mxn<float,static_cast<int64_t>(4L),static_cast<int64_t>(8)>(tmp11, static_cast<int64_t>(8), out_ptr0 + static_cast<int64_t>(x2 + (12L*x1) + (144L*x3) + (9216L*x0)), static_cast<int64_t>(144L));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_2 = async_compile.cpp_pybinding(['float*', 'const int64_t'], '''
#include "/tmp/torchinductor_root/tmptex2y7qd/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const int64_t ks0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L*ks0); x0+=static_cast<int64_t>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
            }
        }
    }
}
''')


cpp_fused__log_softmax_3 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*', 'const int64_t'], '''
#include "/tmp/torchinductor_root/tmptex2y7qd/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const int64_t ks0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(8));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                for(int64_t x1=static_cast<int64_t>(8L); x1<static_cast<int64_t>(10L); x1+=static_cast<int64_t>(2L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
                    tmp_acc0_vec = max_masked_reduce(tmp_acc0_vec, tmp0, static_cast<int64_t>(2L));
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(8));
                    auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                for(int64_t x1=static_cast<int64_t>(8L); x1<static_cast<int64_t>(10L); x1+=static_cast<int64_t>(2L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
                    auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp4, static_cast<int64_t>(2L));
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(8));
                auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                auto tmp4 = out_ptr1[static_cast<int64_t>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = std::log(tmp4);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp3 - tmp6;
                tmp7.store(out_ptr2 + static_cast<int64_t>(x1 + (10L*x0)));
            }
            for(int64_t x1=static_cast<int64_t>(8L); x1<static_cast<int64_t>(10L); x1+=static_cast<int64_t>(2L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
                auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                auto tmp4 = out_ptr1[static_cast<int64_t>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = std::log(tmp4);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp3 - tmp6;
                tmp7.store(out_ptr2 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    s0 = arg2_1
    assert_size_stride(arg0_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (s0, 1, 28, 28), (784, 784, 28, 1))
    assert_size_stride(arg4_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (128, 9216), (9216, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (10, 128), (128, 1))
    assert_size_stride(arg9_1, (10, ), (1, ))
    # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
    buf0 = extern_kernels.convolution(arg3_1, arg0_1, arg1_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf0, (s0, 32, 26, 26), (21632, 676, 26, 1))
    del arg0_1
    del arg1_1
    del arg3_1
    buf1 = empty_strided_cpu((s0, 32, 26, 26), (21632, 1, 832, 32), torch.float32)
    buf2 = empty_strided_cpu((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
    cpp_fused_convolution_relu_0(buf0, arg4_1, buf1, buf2, s0)
    del arg4_1
    del buf0
    # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.relu, aten.convolution]
    buf3 = extern_kernels.convolution(buf1, buf2, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf3, (s0, 64, 24, 24), (36864, 1, 1536, 64))
    del arg5_1
    del buf1
    del buf2
    buf4 = empty_strided_cpu((s0, 64, 12, 12), (9216, 144, 12, 1), torch.float32)
    cpp_fused_max_pool2d_with_indices_relu_1(buf3, buf4, s0)
    del buf3
    buf5 = empty_strided_cpu((s0, 128), (128, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf4, (s0, 9216), (9216, 1), 0), reinterpret_tensor(arg6_1, (9216, 128), (1, 9216), 0), alpha=1, beta=1, out=buf5)
    del arg6_1
    del arg7_1
    del buf4
    buf6 = buf5; del buf5  # reuse
    cpp_fused_relu_2(buf6, s0)
    buf7 = empty_strided_cpu((s0, 10), (10, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_8, x_10], Original ATen: [aten.relu, aten.addmm]
    extern_kernels.addmm(arg9_1, buf6, reinterpret_tensor(arg8_1, (128, 10), (1, 128), 0), alpha=1, beta=1, out=buf7)
    del arg8_1
    del arg9_1
    del buf6
    buf8 = empty_strided_cpu((s0, 1), (1, s0), torch.float32)
    buf9 = empty_strided_cpu((s0, 1), (1, s0), torch.float32)
    buf10 = empty_strided_cpu((s0, 10), (10, 1), torch.float32)
    cpp_fused__log_softmax_3(buf7, buf8, buf9, buf10, s0)
    return (buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = 1000
    arg3_1 = rand_strided((1000, 1, 28, 28), (784, 784, 28, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((128, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((10, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
