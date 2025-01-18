
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1):
        convolution = torch.ops.aten.convolution.default(arg3_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg3_1 = arg0_1 = arg1_1 = None
        relu = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1 = torch.ops.aten.convolution.default(relu, arg4_1, arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg4_1 = arg5_1 = None
        relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool2d_with_offsets = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem = _low_memory_max_pool2d_with_offsets[0];  _low_memory_max_pool2d_with_offsets = None
        view = torch.ops.aten.view.default(getitem, [arg2_1, 9216]);  getitem = arg2_1 = None
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        relu_2 = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, relu_2, permute_1);  arg9_1 = relu_2 = permute_1 = None
        amax = torch.ops.aten.amax.default(addmm_1, [1], True)
        sub_12 = torch.ops.aten.sub.Tensor(addmm_1, amax);  addmm_1 = amax = None
        exp = torch.ops.aten.exp.default(sub_12)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_13 = torch.ops.aten.sub.Tensor(sub_12, log);  sub_12 = log = None
        return (sub_13,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1152)
    reader.tensor(buf0, (32, 1, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 128)
    reader.tensor(buf1, (32,), is_leaf=True)  # arg1_1
    reader.symint(1000)  # arg2_1
    buf2 = reader.storage(None, 3136*s0)
    reader.tensor(buf2, (s0, 1, 28, 28), is_leaf=True)  # arg3_1
    buf3 = reader.storage(None, 73728)
    reader.tensor(buf3, (64, 32, 3, 3), is_leaf=True)  # arg4_1
    buf4 = reader.storage(None, 256)
    reader.tensor(buf4, (64,), is_leaf=True)  # arg5_1
    buf5 = reader.storage(None, 4718592)
    reader.tensor(buf5, (128, 9216), is_leaf=True)  # arg6_1
    buf6 = reader.storage(None, 512)
    reader.tensor(buf6, (128,), is_leaf=True)  # arg7_1
    buf7 = reader.storage(None, 5120)
    reader.tensor(buf7, (10, 128), is_leaf=True)  # arg8_1
    buf8 = reader.storage(None, 40)
    reader.tensor(buf8, (10,), is_leaf=True)  # arg9_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)