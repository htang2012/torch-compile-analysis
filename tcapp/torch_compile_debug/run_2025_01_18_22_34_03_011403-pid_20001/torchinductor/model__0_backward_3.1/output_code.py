# AOT ID: ['0_backward']
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


cpp_fused__log_softmax_backward_data_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_root/tmpr_51oeqw/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(8));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                for(int64_t x1=static_cast<int64_t>(8L); x1<static_cast<int64_t>(10L); x1+=static_cast<int64_t>(2L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
                    tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp0, static_cast<int64_t>(2L));
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(8));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(8));
                auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                auto tmp2 = tmp1.exp();
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 - tmp5;
                tmp6.store(out_ptr1 + static_cast<int64_t>(x1 + (10L*x0)));
            }
            for(int64_t x1=static_cast<int64_t>(8L); x1<static_cast<int64_t>(10L); x1+=static_cast<int64_t>(2L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
                auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                auto tmp2 = tmp1.exp();
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 - tmp5;
                tmp6.store(out_ptr1 + static_cast<int64_t>(x1 + (10L*x0)), static_cast<int64_t>(2L));
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_sum_threshold_backward_1 = async_compile.cpp_pybinding(['float*', 'const float*', 'const bool*', 'const bool*', 'float*'], '''
#include "/tmp/torchinductor_root/tmpr_51oeqw/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + (10L*x1)), static_cast<int64_t>(8));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
            }
        }
        for(int64_t x0=static_cast<int64_t>(8L); x0<static_cast<int64_t>(10L); x0+=static_cast<int64_t>(2L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + (10L*x1)), static_cast<int64_t>(2L));
                    tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp0, static_cast<int64_t>(2L));
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(2L));
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8192L); x0+=static_cast<int64_t>(8L))
        {
            auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr1 + static_cast<int64_t>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
            auto tmp2 = at::vec::VecMask<float,1>::from(in_ptr2 + static_cast<int64_t>(x0));
            auto tmp3 = tmp2.to<float,1>();
            auto tmp4 = static_cast<float>(2.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = tmp1 * tmp6;
            auto tmp8 = static_cast<float>(0.0);
            auto tmp9 = at::vec::Vectorized<float>(tmp8);
            auto tmp10 = decltype(tmp9)::blendv(tmp7, tmp9, tmp0.template cast<float,1>());
            tmp10.store(in_out_ptr0 + static_cast<int64_t>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_max_pool2d_with_indices_max_pool2d_with_indices_backward_native_dropout_backward_sum_threshold_backward_2 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const bool*', 'const int8_t*', 'const float*', 'float*'], '''
#include "/tmp/torchinductor_root/tmpr_51oeqw/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const int8_t* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(8L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + (128L*x1)), static_cast<int64_t>(8));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(589824L); x0+=static_cast<int64_t>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                auto tmp1 = at::vec::VecMask<float,1>::from(in_ptr1 + static_cast<int64_t>(x0));
                auto tmp2 = tmp1.to<float,1>();
                auto tmp3 = static_cast<float>(1.3333333333333333);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 * tmp5;
                tmp6.store(in_out_ptr0 + static_cast<int64_t>(x0));
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(24L); x1+=static_cast<int64_t>(1L))
                {
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(24L); x2+=static_cast<int64_t>(1L))
                    {
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(64L); x3+=static_cast<int64_t>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<int8_t>::loadu(in_ptr2 + static_cast<int64_t>(x3 + (64L*(std::min(static_cast<int64_t>(std::max(static_cast<int64_t>(0L), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(2L))))), static_cast<int64_t>((-1L) + (std::min(static_cast<int64_t>(12L), static_cast<int64_t>(1L + (c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(2L)))))))))) + (768L*(std::min(static_cast<int64_t>(std::max(static_cast<int64_t>(0L), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L))))), static_cast<int64_t>((-1L) + (std::min(static_cast<int64_t>(12L), static_cast<int64_t>(1L + (c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L)))))))))) + (9216L*x0)), static_cast<int64_t>(8));
                            auto tmp21 =
                            [&]
                            {
                                __at_align__ std::array<float, 8> tmpbuf;
                                #pragma GCC unroll 8
                                for (long x3_inner = 0; x3_inner < static_cast<int64_t>(8); x3_inner++)
                                {
                                    tmpbuf[x3_inner] = in_out_ptr0[static_cast<int64_t>((12L*(std::min(static_cast<int64_t>(std::max(static_cast<int64_t>(0L), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L))))), static_cast<int64_t>((-1L) + (std::min(static_cast<int64_t>(12L), static_cast<int64_t>(1L + (c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L)))))))))) + (144L*x3) + (144L*x3_inner) + (9216L*x0) + (std::min(static_cast<int64_t>(std::max(static_cast<int64_t>(0L), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(2L))))), static_cast<int64_t>((-1L) + (std::min(static_cast<int64_t>(12L), static_cast<int64_t>(1L + (c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(2L))))))))))];
                                }
                                return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                            }
                            ()
                            ;
                            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x3 + (64L*x2) + (1536L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                            auto tmp1 = static_cast<int32_t>(2);
                            auto tmp2 = at::vec::convert<int32_t>(tmp0);
                            auto tmp3 = at::vec::Vectorized<int32_t>(tmp1);
                            auto tmp4 = decltype(tmp2)::blendv(tmp2 / tmp3, tmp2 / tmp3 - decltype(tmp2)(1), (tmp2 % tmp3 != decltype(tmp2)(0)) & ((tmp2 < decltype(tmp2)(0)) != (tmp3 < decltype(tmp2)(0))));
                            auto tmp5 = tmp4 * tmp3;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = 2L*(std::min(static_cast<int64_t>(std::max(static_cast<int64_t>(0L), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L))))), static_cast<int64_t>((-1L) + (std::min(static_cast<int64_t>(12L), static_cast<int64_t>(1L + (c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L)))))))));
                            auto tmp8 = c10::convert<int64_t>(tmp7);
                            auto tmp9 = at::vec::convert<int64_t,2,int32_t,1>(tmp4);
                            auto tmp10 = at::vec::VectorizedN<int64_t,2>(tmp8);
                            auto tmp11 = tmp10 + tmp9;
                            auto tmp12 = 2L*(std::min(static_cast<int64_t>(std::max(static_cast<int64_t>(0L), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(2L))))), static_cast<int64_t>((-1L) + (std::min(static_cast<int64_t>(12L), static_cast<int64_t>(1L + (c10::div_floor_integer(static_cast<int64_t>(x2), static_cast<int64_t>(2L)))))))));
                            auto tmp13 = c10::convert<int64_t>(tmp12);
                            auto tmp14 = at::vec::convert<int64_t,2,int32_t,1>(tmp6);
                            auto tmp15 = at::vec::VectorizedN<int64_t,2>(tmp13);
                            auto tmp16 = tmp15 + tmp14;
                            auto tmp17 = static_cast<int64_t>(24);
                            auto tmp18 = at::vec::VectorizedN<int64_t,2>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp20 = tmp19 + tmp16;
                            auto tmp22 = x2 + (24L*x1);
                            auto tmp23 = c10::convert<int32_t>(tmp22);
                            auto tmp24 = c10::convert<int64_t>(tmp23);
                            auto tmp25 = at::vec::VectorizedN<int64_t,2>(tmp24);
                            auto tmp26 = at::vec::VecMask<int64_t,2>(tmp20 == tmp25);
                            auto tmp27 = static_cast<float>(0.0);
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = decltype(tmp21)::blendv(tmp28, tmp21, tmp26.template cast<float,1>());
                            auto tmp31 = at::vec::VecMask<float,1>(tmp30 <= tmp28);
                            auto tmp32 = decltype(tmp28)::blendv(tmp29, tmp28, tmp31.template cast<float,1>());
                            tmp32.store(in_out_ptr1 + static_cast<int64_t>(x3 + (64L*x2) + (1536L*x1) + (36864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_3 = async_compile.cpp_pybinding(['float*', 'const float*'], '''
#include "/tmp/torchinductor_root/tmpr_51oeqw/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(1384448L); x0+=static_cast<int64_t>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3.template cast<float,1>());
                tmp5.store(in_out_ptr0 + static_cast<int64_t>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_4, relu, relu_1, getitem_1, gt, view, gt_1, mul_3, sub_1, permute_2, le, permute_6, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_3, (64, 1, 28, 28), (784, 784, 28, 1))
    assert_size_stride(primals_4, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(relu, (64, 32, 26, 26), (21632, 1, 832, 32))
    assert_size_stride(relu_1, (64, 64, 24, 24), (36864, 1, 1536, 64))
    assert_size_stride(getitem_1, (64, 64, 12, 12), (9216, 1, 768, 64))
    assert_size_stride(gt, (64, 64, 12, 12), (9216, 144, 12, 1))
    assert_size_stride(view, (64, 9216), (9216, 1))
    assert_size_stride(gt_1, (64, 128), (128, 1))
    assert_size_stride(mul_3, (64, 128), (128, 1))
    assert_size_stride(sub_1, (64, 10), (10, 1))
    assert_size_stride(permute_2, (10, 128), (128, 1))
    assert_size_stride(le, (64, 128), (128, 1))
    assert_size_stride(permute_6, (128, 9216), (9216, 1))
    assert_size_stride(tangents_1, (64, 10), (10, 1))
    buf0 = empty_strided_cpu((64, 1), (1, 64), torch.float32)
    buf1 = empty_strided_cpu((64, 10), (10, 1), torch.float32)
    cpp_fused__log_softmax_backward_data_0(tangents_1, sub_1, buf0, buf1)
    del buf0
    del sub_1
    del tangents_1
    buf2 = empty_strided_cpu((64, 128), (128, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf1, permute_2, out=buf2)
    del permute_2
    buf3 = empty_strided_cpu((10, 128), (128, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (10, 64), (1, 10), 0), mul_3, out=buf3)
    del mul_3
    buf4 = empty_strided_cpu((1, 10), (10, 1), torch.float32)
    buf5 = buf2; del buf2  # reuse
    cpp_fused_native_dropout_backward_sum_threshold_backward_1(buf5, buf1, le, gt_1, buf4)
    del buf1
    del gt_1
    del le
    buf6 = empty_strided_cpu((64, 9216), (9216, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, permute_6, out=buf6)
    del permute_6
    buf7 = empty_strided_cpu((128, 9216), (9216, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (128, 64), (1, 128), 0), view, out=buf7)
    del view
    buf8 = empty_strided_cpu((1, 128), (128, 1), torch.float32)
    buf9 = reinterpret_tensor(buf6, (64, 64, 12, 12), (9216, 144, 12, 1), 0); del buf6  # reuse
    buf10 = empty_strided_cpu((64, 64, 24, 24), (36864, 1, 1536, 64), torch.float32)
    buf11 = buf10; del buf10  # reuse
    cpp_fused_convolution_backward_max_pool2d_with_indices_max_pool2d_with_indices_backward_native_dropout_backward_sum_threshold_backward_2(buf9, buf11, buf5, gt, getitem_1, relu_1, buf8)
    del buf5
    del buf9
    del getitem_1
    del gt
    del relu_1
    # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
    buf12 = torch.ops.aten.convolution_backward.default(buf11, relu, primals_4, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf11
    del primals_4
    buf13 = buf12[0]
    buf14 = buf12[1]
    buf15 = buf12[2]
    del buf12
    buf16 = buf13; del buf13  # reuse
    cpp_fused_convolution_backward_threshold_backward_3(buf16, relu)
    del relu
    # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
    buf17 = torch.ops.aten.convolution_backward.default(buf16, primals_3, primals_1, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf16
    del primals_1
    del primals_3
    buf18 = buf17[1]
    buf19 = buf17[2]
    return (buf18, buf19, None, buf14, buf15, buf7, reinterpret_tensor(buf8, (128, ), (1, ), 0), buf3, reinterpret_tensor(buf4, (10, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, 1, 28, 28), (784, 784, 28, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    relu = rand_strided((64, 32, 26, 26), (21632, 1, 832, 32), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((64, 64, 24, 24), (36864, 1, 1536, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((64, 64, 12, 12), (9216, 1, 768, 64), device='cpu', dtype=torch.int8)
    gt = rand_strided((64, 64, 12, 12), (9216, 144, 12, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((64, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    gt_1 = rand_strided((64, 128), (128, 1), device='cpu', dtype=torch.bool)
    mul_3 = rand_strided((64, 128), (128, 1), device='cpu', dtype=torch.float32)
    sub_1 = rand_strided((64, 10), (10, 1), device='cpu', dtype=torch.float32)
    permute_2 = rand_strided((10, 128), (128, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((64, 128), (128, 1), device='cpu', dtype=torch.bool)
    permute_6 = rand_strided((128, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((64, 10), (10, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_3, primals_4, relu, relu_1, getitem_1, gt, view, gt_1, mul_3, sub_1, permute_2, le, permute_6, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
