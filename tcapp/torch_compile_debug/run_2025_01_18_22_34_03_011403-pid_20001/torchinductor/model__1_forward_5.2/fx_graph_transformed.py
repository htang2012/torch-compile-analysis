class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 1, 3, 3]", primals_2: "f32[32]", primals_3: "Sym(s0)", primals_4: "f32[s0, 1, 28, 28]", primals_5: "f32[64, 32, 3, 3]", primals_6: "f32[64]", primals_7: "f32[128, 9216]", primals_8: "f32[128]", primals_9: "f32[10, 128]", primals_10: "f32[10]"):
         # File: /app/tcapp/main_2.py:89 in forward, code: x = self.conv1(x)
        convolution: "f32[s0, 32, 26, 26]" = torch.ops.aten.convolution.default(primals_4, primals_1, primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
        
         # File: /app/tcapp/main_2.py:90 in forward, code: x = F.relu(x)
        relu: "f32[s0, 32, 26, 26]" = torch.ops.aten.relu.default(convolution);  convolution = None
        
         # File: /app/tcapp/main_2.py:91 in forward, code: x = self.conv2(x)
        convolution_1: "f32[s0, 64, 24, 24]" = torch.ops.aten.convolution.default(relu, primals_5, primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
        
         # File: /app/tcapp/main_2.py:92 in forward, code: x = F.relu(x)
        relu_1: "f32[s0, 64, 24, 24]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        
         # File: /app/tcapp/main_2.py:93 in forward, code: x = F.max_pool2d(x, 2)
        _low_memory_max_pool2d_with_offsets = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem: "f32[s0, 64, 12, 12]" = _low_memory_max_pool2d_with_offsets[0]
        getitem_1: "i8[s0, 64, 12, 12]" = _low_memory_max_pool2d_with_offsets[1];  _low_memory_max_pool2d_with_offsets = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cpu'))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[s0, 64, 12, 12]" = torch.ops.prims.inductor_random.default([primals_3, 64, 12, 12], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        
         # File: /app/tcapp/main_2.py:94 in forward, code: x = self.dropout1(x)
        gt: "b8[s0, 64, 12, 12]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.25);  inductor_random_default_1 = None
        mul_16: "f32[s0, 64, 12, 12]" = torch.ops.aten.mul.Tensor(gt, getitem);  getitem = None
        mul_17: "f32[s0, 64, 12, 12]" = torch.ops.aten.mul.Tensor(mul_16, 1.3333333333333333);  mul_16 = None
        
         # File: /app/tcapp/main_2.py:95 in forward, code: x = torch.flatten(x, 1)
        view: "f32[s0, 9216]" = torch.ops.aten.reshape.default(mul_17, [primals_3, 9216]);  mul_17 = None
        
         # File: /app/tcapp/main_2.py:96 in forward, code: x = self.fc1(x)
        permute: "f32[9216, 128]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm: "f32[s0, 128]" = torch.ops.aten.addmm.default(primals_8, view, permute);  primals_8 = None
        
         # File: /app/tcapp/main_2.py:97 in forward, code: x = F.relu(x)
        relu_2: "f32[s0, 128]" = torch.ops.aten.relu.default(addmm);  addmm = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[s0, 128]" = torch.ops.prims.inductor_random.default([primals_3, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        
         # File: /app/tcapp/main_2.py:98 in forward, code: x = self.dropout2(x)
        gt_1: "b8[s0, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_33: "f32[s0, 128]" = torch.ops.aten.mul.Tensor(gt_1, relu_2)
        mul_34: "f32[s0, 128]" = torch.ops.aten.mul.Tensor(mul_33, 2.0);  mul_33 = None
        
         # File: /app/tcapp/main_2.py:99 in forward, code: x = self.fc2(x)
        permute_1: "f32[128, 10]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        addmm_1: "f32[s0, 10]" = torch.ops.aten.addmm.default(primals_10, mul_34, permute_1);  primals_10 = None
        
         # File: /app/tcapp/main_2.py:100 in forward, code: output = F.log_softmax(x, dim=1)
        amax: "f32[s0, 1]" = torch.ops.aten.amax.default(addmm_1, [1], True)
        sub_17: "f32[s0, 10]" = torch.ops.aten.sub.Tensor(addmm_1, amax);  addmm_1 = amax = None
        exp: "f32[s0, 10]" = torch.ops.aten.exp.default(sub_17)
        sum_1: "f32[s0, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[s0, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_18: "f32[s0, 10]" = torch.ops.aten.sub.Tensor(sub_17, log);  sub_17 = log = None
        
         # File: /app/tcapp/main_2.py:99 in forward, code: x = self.fc2(x)
        permute_2: "f32[10, 128]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /app/tcapp/main_2.py:97 in forward, code: x = F.relu(x)
        le: "b8[s0, 128]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        
         # File: /app/tcapp/main_2.py:96 in forward, code: x = self.fc1(x)
        permute_6: "f32[128, 9216]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (sub_18, primals_1, primals_4, primals_5, relu, relu_1, getitem_1, gt, view, gt_1, mul_34, sub_18, permute_2, le, permute_6, primals_3)
        