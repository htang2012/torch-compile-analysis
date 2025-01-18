class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 1, 3, 3]", primals_3: "f32[64, 1, 28, 28]", primals_4: "f32[64, 32, 3, 3]", relu: "f32[64, 32, 26, 26]", relu_1: "f32[64, 64, 24, 24]", getitem_1: "i8[64, 64, 12, 12]", gt: "b8[64, 64, 12, 12]", view: "f32[64, 9216]", gt_1: "b8[64, 128]", mul_3: "f32[64, 128]", sub_1: "f32[64, 10]", permute_2: "f32[10, 128]", le: "b8[64, 128]", permute_6: "f32[128, 9216]", tangents_1: "f32[64, 10]"):
         # File: /app/tcapp/main_2.py:100 in forward, code: output = F.log_softmax(x, dim=1)
        sum_2: "f32[64, 1]" = torch.ops.aten.sum.dim_IntList(tangents_1, [1], True)
        exp_1: "f32[64, 10]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul_4: "f32[64, 10]" = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2: "f32[64, 10]" = torch.ops.aten.sub.Tensor(tangents_1, mul_4);  tangents_1 = mul_4 = None
        
         # File: /app/tcapp/main_2.py:99 in forward, code: x = self.fc2(x)
        mm: "f32[64, 128]" = torch.ops.aten.mm.default(sub_2, permute_2);  permute_2 = None
        permute_3: "f32[10, 64]" = torch.ops.aten.permute.default(sub_2, [1, 0])
        mm_1: "f32[10, 128]" = torch.ops.aten.mm.default(permute_3, mul_3);  permute_3 = mul_3 = None
        permute_4: "f32[128, 10]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_3: "f32[1, 10]" = torch.ops.aten.sum.dim_IntList(sub_2, [0], True);  sub_2 = None
        view_1: "f32[10]" = torch.ops.aten.reshape.default(sum_3, [10]);  sum_3 = None
        permute_5: "f32[10, 128]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
         # File: /app/tcapp/main_2.py:98 in forward, code: x = self.dropout2(x)
        convert_element_type: "f32[64, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_5: "f32[64, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_6: "f32[64, 128]" = torch.ops.aten.mul.Tensor(mm, mul_5);  mm = mul_5 = None
        
         # File: /app/tcapp/main_2.py:97 in forward, code: x = F.relu(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: "f32[64, 128]" = torch.ops.aten.where.self(le, full_default, mul_6);  le = mul_6 = None
        
         # File: /app/tcapp/main_2.py:96 in forward, code: x = self.fc1(x)
        mm_2: "f32[64, 9216]" = torch.ops.aten.mm.default(where, permute_6);  permute_6 = None
        permute_7: "f32[128, 64]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "f32[128, 9216]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        permute_8: "f32[9216, 128]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_4: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_2: "f32[128]" = torch.ops.aten.reshape.default(sum_4, [128]);  sum_4 = None
        permute_9: "f32[128, 9216]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
         # File: /app/tcapp/main_2.py:95 in forward, code: x = torch.flatten(x, 1)
        view_3: "f32[64, 64, 12, 12]" = torch.ops.aten.reshape.default(mm_2, [64, 64, 12, 12]);  mm_2 = None
        
         # File: /app/tcapp/main_2.py:94 in forward, code: x = self.dropout1(x)
        convert_element_type_1: "f32[64, 64, 12, 12]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_7: "f32[64, 64, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.3333333333333333);  convert_element_type_1 = None
        mul_8: "f32[64, 64, 12, 12]" = torch.ops.aten.mul.Tensor(view_3, mul_7);  view_3 = mul_7 = None
        
         # File: /app/tcapp/main_2.py:93 in forward, code: x = F.max_pool2d(x, 2)
        _low_memory_max_pool2d_offsets_to_indices: "i64[64, 64, 12, 12]" = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_1, 2, 24, [2, 2], [0, 0]);  getitem_1 = None
        max_pool2d_with_indices_backward: "f32[64, 64, 24, 24]" = torch.ops.aten.max_pool2d_with_indices_backward.default(mul_8, relu_1, [2, 2], [], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices);  mul_8 = _low_memory_max_pool2d_offsets_to_indices = None
        
         # File: /app/tcapp/main_2.py:92 in forward, code: x = F.relu(x)
        le_1: "b8[64, 64, 24, 24]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_1: "f32[64, 64, 24, 24]" = torch.ops.aten.where.self(le_1, full_default, max_pool2d_with_indices_backward);  le_1 = max_pool2d_with_indices_backward = None
        
         # File: /app/tcapp/main_2.py:91 in forward, code: x = self.conv2(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(where_1, relu, primals_4, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = primals_4 = None
        getitem_2: "f32[64, 32, 26, 26]" = convolution_backward[0]
        getitem_3: "f32[64, 32, 3, 3]" = convolution_backward[1]
        getitem_4: "f32[64]" = convolution_backward[2];  convolution_backward = None
        
         # File: /app/tcapp/main_2.py:90 in forward, code: x = F.relu(x)
        le_2: "b8[64, 32, 26, 26]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_2: "f32[64, 32, 26, 26]" = torch.ops.aten.where.self(le_2, full_default, getitem_2);  le_2 = full_default = getitem_2 = None
        
         # File: /app/tcapp/main_2.py:89 in forward, code: x = self.conv1(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_2, primals_3, primals_1, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  where_2 = primals_3 = primals_1 = None
        getitem_6: "f32[32, 1, 3, 3]" = convolution_backward_1[1]
        getitem_7: "f32[32]" = convolution_backward_1[2];  convolution_backward_1 = None
        return (getitem_6, getitem_7, None, getitem_3, getitem_4, permute_9, view_2, permute_5, view_1)
        