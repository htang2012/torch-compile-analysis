class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 1, 3, 3]", arg1_1: "f32[32]", arg2_1: "Sym(s0)", arg3_1: "f32[s0, 1, 28, 28]", arg4_1: "f32[64, 32, 3, 3]", arg5_1: "f32[64]", arg6_1: "f32[128, 9216]", arg7_1: "f32[128]", arg8_1: "f32[10, 128]", arg9_1: "f32[10]"):
         # File: /app/tcapp/main_2.py:89 in forward, code: x = self.conv1(x)
        convolution: "f32[s0, 32, 26, 26]" = torch.ops.aten.convolution.default(arg3_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg3_1 = arg0_1 = arg1_1 = None
        
         # File: /app/tcapp/main_2.py:90 in forward, code: x = F.relu(x)
        relu: "f32[s0, 32, 26, 26]" = torch.ops.aten.relu.default(convolution);  convolution = None
        
         # File: /app/tcapp/main_2.py:91 in forward, code: x = self.conv2(x)
        convolution_1: "f32[s0, 64, 24, 24]" = torch.ops.aten.convolution.default(relu, arg4_1, arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu = arg4_1 = arg5_1 = None
        
         # File: /app/tcapp/main_2.py:92 in forward, code: x = F.relu(x)
        relu_1: "f32[s0, 64, 24, 24]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        
         # File: /app/tcapp/main_2.py:93 in forward, code: x = F.max_pool2d(x, 2)
        _low_memory_max_pool2d_with_offsets = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem: "f32[s0, 64, 12, 12]" = _low_memory_max_pool2d_with_offsets[0];  _low_memory_max_pool2d_with_offsets = None
        
         # File: /app/tcapp/main_2.py:95 in forward, code: x = torch.flatten(x, 1)
        view: "f32[s0, 9216]" = torch.ops.aten.view.default(getitem, [arg2_1, 9216]);  getitem = arg2_1 = None
        
         # File: /app/tcapp/main_2.py:96 in forward, code: x = self.fc1(x)
        permute: "f32[9216, 128]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "f32[s0, 128]" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        
         # File: /app/tcapp/main_2.py:97 in forward, code: x = F.relu(x)
        relu_2: "f32[s0, 128]" = torch.ops.aten.relu.default(addmm);  addmm = None
        
         # File: /app/tcapp/main_2.py:99 in forward, code: x = self.fc2(x)
        permute_1: "f32[128, 10]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "f32[s0, 10]" = torch.ops.aten.addmm.default(arg9_1, relu_2, permute_1);  arg9_1 = relu_2 = permute_1 = None
        
         # File: /app/tcapp/main_2.py:100 in forward, code: output = F.log_softmax(x, dim=1)
        amax: "f32[s0, 1]" = torch.ops.aten.amax.default(addmm_1, [1], True)
        sub_12: "f32[s0, 10]" = torch.ops.aten.sub.Tensor(addmm_1, amax);  addmm_1 = amax = None
        exp: "f32[s0, 10]" = torch.ops.aten.exp.default(sub_12)
        sum_1: "f32[s0, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[s0, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_13: "f32[s0, 10]" = torch.ops.aten.sub.Tensor(sub_12, log);  sub_12 = log = None
        return (sub_13,)
        