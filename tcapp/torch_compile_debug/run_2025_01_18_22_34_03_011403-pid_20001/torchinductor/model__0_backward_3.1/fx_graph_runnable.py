
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.debug = True




isolate_fails_code_str = None



# torch version: 2.5.1+cu124
# torch cuda version: 12.4
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_3, primals_4, relu, relu_1, getitem_1, gt, view, gt_1, mul_3, sub_1, permute_2, le, permute_6, tangents_1):
        sum_2 = torch.ops.aten.sum.dim_IntList(tangents_1, [1], True)
        exp_1 = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(tangents_1, mul_4);  tangents_1 = mul_4 = None
        mm = torch.ops.aten.mm.default(sub_2, permute_2);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(sub_2, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_3, mul_3);  permute_3 = mul_3 = None
        permute_4 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(sub_2, [0], True);  sub_2 = None
        view_1 = torch.ops.aten.view.default(sum_3, [10]);  sum_3 = None
        permute_5 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_6 = torch.ops.aten.mul.Tensor(mm, mul_5);  mm = mul_5 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, mul_6);  le = mul_6 = None
        mm_2 = torch.ops.aten.mm.default(where, permute_6);  permute_6 = None
        permute_7 = torch.ops.aten.permute.default(where, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        permute_8 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_2 = torch.ops.aten.view.default(sum_4, [128]);  sum_4 = None
        permute_9 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        view_3 = torch.ops.aten.view.default(mm_2, [64, 64, 12, 12]);  mm_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.3333333333333333);  convert_element_type_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(view_3, mul_7);  view_3 = mul_7 = None
        _low_memory_max_pool2d_offsets_to_indices = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_1, 2, 24, [2, 2], [0, 0]);  getitem_1 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(mul_8, relu_1, [2, 2], [], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices);  mul_8 = _low_memory_max_pool2d_offsets_to_indices = None
        le_1 = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_1 = torch.ops.aten.where.self(le_1, full_default, max_pool2d_with_indices_backward);  le_1 = max_pool2d_with_indices_backward = None
        convolution_backward = torch.ops.aten.convolution_backward.default(where_1, relu, primals_4, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = primals_4 = None
        getitem_2 = convolution_backward[0]
        getitem_3 = convolution_backward[1]
        getitem_4 = convolution_backward[2];  convolution_backward = None
        le_2 = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_2 = torch.ops.aten.where.self(le_2, full_default, getitem_2);  le_2 = full_default = getitem_2 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_2, primals_3, primals_1, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  where_2 = primals_3 = primals_1 = None
        getitem_6 = convolution_backward_1[1]
        getitem_7 = convolution_backward_1[2];  convolution_backward_1 = None
        return (getitem_6, getitem_7, None, getitem_3, getitem_4, permute_9, view_2, permute_5, view_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 1152)
    reader.tensor(buf0, (32, 1, 3, 3), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 200704)
    reader.tensor(buf1, (64, 1, 28, 28), is_leaf=True)  # primals_3
    buf2 = reader.storage(None, 73728)
    reader.tensor(buf2, (64, 32, 3, 3), (288, 1, 96, 32), is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 5537792)
    reader.tensor(buf3, (64, 32, 26, 26), (21632, 1, 832, 32), is_leaf=True)  # relu
    buf4 = reader.storage(None, 9437184)
    reader.tensor(buf4, (64, 64, 24, 24), (36864, 1, 1536, 64), is_leaf=True)  # relu_1
    buf5 = reader.storage(None, 589824, dtype_hint=torch.int8)
    reader.tensor(buf5, (64, 64, 12, 12), (9216, 1, 768, 64), dtype=torch.int8, is_leaf=True)  # getitem_1
    buf6 = reader.storage(None, 589824, dtype_hint=torch.bool)
    reader.tensor(buf6, (64, 64, 12, 12), dtype=torch.bool, is_leaf=True)  # gt
    buf7 = reader.storage(None, 2359296)
    reader.tensor(buf7, (64, 9216), is_leaf=True)  # view
    buf8 = reader.storage(None, 8192, dtype_hint=torch.bool)
    reader.tensor(buf8, (64, 128), dtype=torch.bool, is_leaf=True)  # gt_1
    buf9 = reader.storage(None, 32768)
    reader.tensor(buf9, (64, 128), is_leaf=True)  # mul_3
    buf10 = reader.storage(None, 2560)
    reader.tensor(buf10, (64, 10), is_leaf=True)  # sub_1
    buf11 = reader.storage(None, 5120)
    reader.tensor(buf11, (10, 128), is_leaf=True)  # permute_2
    buf12 = reader.storage(None, 8192, dtype_hint=torch.bool)
    reader.tensor(buf12, (64, 128), dtype=torch.bool, is_leaf=True)  # le
    buf13 = reader.storage(None, 4718592)
    reader.tensor(buf13, (128, 9216), is_leaf=True)  # permute_6
    buf14 = reader.storage(None, 2560)
    reader.tensor(buf14, (64, 10), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)