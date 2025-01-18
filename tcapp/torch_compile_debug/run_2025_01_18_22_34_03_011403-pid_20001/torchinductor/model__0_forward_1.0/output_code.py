# AOT ID: ['0_forward']
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


cpp_fused_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_root/tmptsztcwo3/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)), static_cast<int64_t>(8));
                            tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp1, static_cast<int64_t>(8), out_ptr0 + static_cast<int64_t>(x1 + (32L*x2) + (288L*x0)), static_cast<int64_t>(32L));
                    }
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(8L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(1L))
                    {
                        alignas(8) float tmp1[8*8];
                        for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)), static_cast<int64_t>(1L));
                            tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                        }
                        at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L)>(tmp1, static_cast<int64_t>(1L), out_ptr0 + static_cast<int64_t>(x1 + (32L*x2) + (288L*x0)), static_cast<int64_t>(32L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_1 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_root/tmptsztcwo3/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
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
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_2 = async_compile.cpp_pybinding(['float*', 'int8_t*'], '''
#include "/tmp/torchinductor_root/tmptsztcwo3/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       int8_t* out_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2359296L); x0+=static_cast<int64_t>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(768L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(12L); x1+=static_cast<int64_t>(1L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x2 + (128L*x1) + (3072L*x0)), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(64L + x2 + (128L*x1) + (3072L*x0)), static_cast<int64_t>(8));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(1536L + x2 + (128L*x1) + (3072L*x0)), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(1600L + x2 + (128L*x1) + (3072L*x0)), static_cast<int64_t>(8));
                        auto tmp2 = at::vec::VecMask<float,1>(tmp1 > tmp0);
                        auto tmp3 = static_cast<int8_t>(1);
                        auto tmp4 = static_cast<int8_t>(0);
                        auto tmp5 = at::vec::Vectorized<int8_t>(tmp3);
                        auto tmp6 = at::vec::Vectorized<int8_t>(tmp4);
                        auto tmp7 = decltype(tmp5)::blendv(tmp6, tmp5, tmp2.template cast<int8_t,1>());
                        auto tmp8 = at::vec::maximum(tmp1, tmp0);
                        auto tmp10 = at::vec::VecMask<float,1>(tmp9 > tmp8);
                        auto tmp11 = static_cast<int8_t>(2);
                        auto tmp12 = at::vec::Vectorized<int8_t>(tmp11);
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp10.template cast<int8_t,1>());
                        auto tmp14 = at::vec::maximum(tmp9, tmp8);
                        auto tmp16 = at::vec::VecMask<float,1>(tmp15 > tmp14);
                        auto tmp17 = static_cast<int8_t>(3);
                        auto tmp18 = at::vec::Vectorized<int8_t>(tmp17);
                        auto tmp19 = decltype(tmp18)::blendv(tmp13, tmp18, tmp16.template cast<int8_t,1>());
                        auto tmp20 = at::vec::maximum(tmp15, tmp14);
                        tmp19.store(out_ptr0 + static_cast<int64_t>(x2 + (64L*x1) + (768L*x0)), static_cast<int64_t>(8));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_max_pool2d_with_indices_native_dropout_3 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'bool*', 'float*'], '''
#include "/tmp/torchinductor_root/tmptsztcwo3/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(589824L); x0+=static_cast<int64_t>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = x0;
                auto tmp2 = c10::convert<int32_t>(tmp1);
                auto tmp3 = at::vec::Vectorized<int32_t>::arange(tmp2, 1);
                auto tmp4 = at::vec::convert<int64_t,2,int32_t,1>(tmp3);
                auto tmp5 =
                [&]()
                {
                    int64_t offset[8];
                    float result[8];
                    tmp4.store(offset);
                    for( int64_t offset_idx = 0; offset_idx < 8; offset_idx++ )
                    {
                        result[offset_idx] = normalized_rand_cpu(tmp0, offset[offset_idx]);
                    }
                    return at::vec::Vectorized<float>::loadu(result);
                }
                ()
                ;
                auto tmp6 = static_cast<float>(0.25);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = at::vec::VecMask<float,1>(tmp5 > tmp7);
                tmp8.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
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
                            alignas(8) bool tmp0[8*8];
                            at::vec::transpose_mxn<bool,static_cast<int64_t>(8),static_cast<int64_t>(8)>(out_ptr1 + static_cast<int64_t>(x2 + (12L*x1) + (144L*x3) + (9216L*x0)), static_cast<int64_t>(144L), tmp0, static_cast<int64_t>(8));
                            alignas(8) float tmp14[8*8];
                            for (long x2_inner = 0; x2_inner < static_cast<int64_t>(8); x2_inner++)
                            {
                                auto tmp1 = at::vec::VecMask<float,1>::from(tmp0 + static_cast<int64_t>(8L*x2_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(64L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(1536L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(1600L + x3 + (128L*x2) + (128L*x2_inner) + (3072L*x1) + (36864L*x0)), static_cast<int64_t>(8));
                                auto tmp2 = tmp1.to<float,1>();
                                auto tmp5 = at::vec::maximum(tmp4, tmp3);
                                auto tmp7 = at::vec::maximum(tmp6, tmp5);
                                auto tmp9 = at::vec::maximum(tmp8, tmp7);
                                auto tmp10 = tmp2 * tmp9;
                                auto tmp11 = static_cast<float>(1.3333333333333333);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp10 * tmp12;
                                tmp13.store(tmp14 + static_cast<int64_t>(8L*x2_inner));
                            }
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp14, static_cast<int64_t>(8), out_ptr2 + static_cast<int64_t>(x2 + (12L*x1) + (144L*x3) + (9216L*x0)), static_cast<int64_t>(144L));
                        }
                    }
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(8L); x2<static_cast<int64_t>(12L); x2+=static_cast<int64_t>(1L))
                    {
                        #pragma GCC ivdep
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(64L); x3+=static_cast<int64_t>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<int64_t>(x2 + (12L*x1) + (144L*x3) + (9216L*x0))];
                            auto tmp2 = in_ptr1[static_cast<int64_t>(x3 + (128L*x2) + (3072L*x1) + (36864L*x0))];
                            auto tmp3 = in_ptr1[static_cast<int64_t>(64L + x3 + (128L*x2) + (3072L*x1) + (36864L*x0))];
                            auto tmp5 = in_ptr1[static_cast<int64_t>(1536L + x3 + (128L*x2) + (3072L*x1) + (36864L*x0))];
                            auto tmp7 = in_ptr1[static_cast<int64_t>(1600L + x3 + (128L*x2) + (3072L*x1) + (36864L*x0))];
                            auto tmp1 = c10::convert<float>(tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp9 = decltype(tmp1)(tmp1 * tmp8);
                            auto tmp10 = static_cast<float>(1.3333333333333333);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            out_ptr2[static_cast<int64_t>(x2 + (12L*x1) + (144L*x3) + (9216L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_dropout_relu_threshold_backward_4 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'bool*', 'float*', 'bool*'], '''
#include "/tmp/torchinductor_root/tmptsztcwo3/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr1,
                       float* out_ptr2,
                       bool* out_ptr3)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8192L); x0+=static_cast<int64_t>(8L))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
            auto tmp1 = x0;
            auto tmp2 = c10::convert<int32_t>(tmp1);
            auto tmp3 = at::vec::Vectorized<int32_t>::arange(tmp2, 1);
            auto tmp4 = at::vec::convert<int64_t,2,int32_t,1>(tmp3);
            auto tmp5 =
            [&]()
            {
                int64_t offset[8];
                float result[8];
                tmp4.store(offset);
                for( int64_t offset_idx = 0; offset_idx < 8; offset_idx++ )
                {
                    result[offset_idx] = normalized_rand_cpu(tmp0, offset[offset_idx]);
                }
                return at::vec::Vectorized<float>::loadu(result);
            }
            ()
            ;
            auto tmp6 = static_cast<float>(0.5);
            auto tmp7 = at::vec::Vectorized<float>(tmp6);
            auto tmp8 = at::vec::VecMask<float,1>(tmp5 > tmp7);
            auto tmp9 = tmp8.to<float,1>();
            auto tmp11 = at::vec::clamp_min(tmp10, decltype(tmp10)(0));
            auto tmp12 = tmp9 * tmp11;
            auto tmp13 = static_cast<float>(2.0);
            auto tmp14 = at::vec::Vectorized<float>(tmp13);
            auto tmp15 = tmp12 * tmp14;
            auto tmp16 = static_cast<float>(0.0);
            auto tmp17 = at::vec::Vectorized<float>(tmp16);
            auto tmp18 = at::vec::VecMask<float,1>(tmp11 <= tmp17);
            tmp8.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
            tmp15.store(out_ptr2 + static_cast<int64_t>(x0));
            tmp18.store(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
        }
    }
}
''')


cpp_fused__log_softmax_5 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_root/tmptsztcwo3/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (64, 1, 28, 28), (784, 784, 28, 1))
    assert_size_stride(primals_4, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (128, 9216), (9216, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (10, 128), (128, 1))
    assert_size_stride(primals_9, (10, ), (1, ))
    buf0 = empty_strided_cpu((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
    cpp_fused_0(primals_4, buf0)
    del primals_4
    # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
    buf1 = extern_kernels.convolution(primals_3, primals_1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf1, (64, 32, 26, 26), (21632, 676, 26, 1))
    del primals_2
    buf2 = empty_strided_cpu((64, 32, 26, 26), (21632, 1, 832, 32), torch.float32)
    cpp_fused_relu_1(buf1, buf2)
    del buf1
    # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
    buf3 = extern_kernels.convolution(buf2, buf0, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf3, (64, 64, 24, 24), (36864, 1, 1536, 64))
    del primals_5
    buf4 = buf3; del buf3  # reuse
    buf5 = empty_strided_cpu((64, 64, 12, 12), (9216, 1, 768, 64), torch.int8)
    cpp_fused_max_pool2d_with_indices_relu_2(buf4, buf5)
    buf6 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf6)
    buf8 = empty_strided_cpu((64, 64, 12, 12), (9216, 144, 12, 1), torch.bool)
    buf9 = empty_strided_cpu((64, 64, 12, 12), (9216, 144, 12, 1), torch.float32)
    cpp_fused_max_pool2d_with_indices_native_dropout_3(buf6, buf4, buf8, buf9)
    buf10 = empty_strided_cpu((64, 128), (128, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_7, reinterpret_tensor(buf9, (64, 9216), (9216, 1), 0), reinterpret_tensor(primals_6, (9216, 128), (1, 9216), 0), alpha=1, beta=1, out=buf10)
    del primals_7
    buf12 = empty_strided_cpu((64, 128), (128, 1), torch.bool)
    buf13 = empty_strided_cpu((64, 128), (128, 1), torch.float32)
    buf18 = empty_strided_cpu((64, 128), (128, 1), torch.bool)
    cpp_fused_native_dropout_relu_threshold_backward_4(buf6, buf10, buf12, buf13, buf18)
    del buf10
    del buf6
    buf14 = empty_strided_cpu((64, 10), (10, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_9, buf13, reinterpret_tensor(primals_8, (128, 10), (1, 128), 0), alpha=1, beta=1, out=buf14)
    del primals_9
    buf15 = empty_strided_cpu((64, 1), (1, 64), torch.float32)
    buf16 = empty_strided_cpu((64, 1), (1, 64), torch.float32)
    buf17 = empty_strided_cpu((64, 10), (10, 1), torch.float32)
    cpp_fused__log_softmax_5(buf14, buf15, buf16, buf17)
    return (buf17, primals_1, primals_3, buf0, buf2, buf4, buf5, buf8, reinterpret_tensor(buf9, (64, 9216), (9216, 1), 0), buf12, buf13, buf17, primals_8, buf18, primals_6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, 1, 28, 28), (784, 784, 28, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((10, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
