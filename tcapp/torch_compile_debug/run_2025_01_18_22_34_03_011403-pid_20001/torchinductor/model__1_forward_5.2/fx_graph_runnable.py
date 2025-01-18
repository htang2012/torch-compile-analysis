
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
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.5.1+cu124
# torch cuda version: 12.4
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10):
        convolution = torch.ops.aten.convolution.default(primals_4, primals_1, primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
        relu = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1 = torch.ops.aten.convolution.default(relu, primals_5, primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
        relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool2d_with_offsets = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem = _low_memory_max_pool2d_with_offsets[0]
        getitem_1 = _low_memory_max_pool2d_with_offsets[1];  _low_memory_max_pool2d_with_offsets = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(2, device(type='cpu'))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([primals_3, 64, 12, 12], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.25);  inductor_random_default_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(gt, getitem);  getitem = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, 1.3333333333333333);  mul_16 = None
        view = torch.ops.aten.view.default(mul_17, [primals_3, 9216]);  mul_17 = None
        permute = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm = torch.ops.aten.addmm.default(primals_8, view, permute);  primals_8 = None
        relu_2 = torch.ops.aten.relu.default(addmm);  addmm = None
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default = torch.ops.prims.inductor_random.default([primals_3, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1 = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_33 = torch.ops.aten.mul.Tensor(gt_1, relu_2)
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, 2.0);  mul_33 = None
        permute_1 = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_10, mul_34, permute_1);  primals_10 = None
        amax = torch.ops.aten.amax.default(addmm_1, [1], True)
        sub_17 = torch.ops.aten.sub.Tensor(addmm_1, amax);  addmm_1 = amax = None
        exp = torch.ops.aten.exp.default(sub_17)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_18 = torch.ops.aten.sub.Tensor(sub_17, log);  sub_17 = log = None
        permute_2 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        le = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        permute_6 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (sub_18, primals_1, primals_4, primals_5, relu, relu_1, getitem_1, gt, view, gt_1, mul_34, sub_18, permute_2, le, permute_6, primals_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 1152)
    reader.tensor(buf0, (32, 1, 3, 3), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 128)
    reader.tensor(buf1, (32,), is_leaf=True)  # primals_2
    reader.symint(32)  # primals_3
    buf2 = reader.storage(None, 3136*s0)
    reader.tensor(buf2, (s0, 1, 28, 28), is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 73728)
    reader.tensor(buf3, (64, 32, 3, 3), is_leaf=True)  # primals_5
    buf4 = reader.storage(None, 256)
    reader.tensor(buf4, (64,), is_leaf=True)  # primals_6
    buf5 = reader.storage(None, 4718592)
    reader.tensor(buf5, (128, 9216), is_leaf=True)  # primals_7
    buf6 = reader.storage(None, 512)
    reader.tensor(buf6, (128,), is_leaf=True)  # primals_8
    buf7 = reader.storage(None, 5120)
    reader.tensor(buf7, (10, 128), is_leaf=True)  # primals_9
    buf8 = reader.storage(None, 40)
    reader.tensor(buf8, (10,), is_leaf=True)  # primals_10
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)