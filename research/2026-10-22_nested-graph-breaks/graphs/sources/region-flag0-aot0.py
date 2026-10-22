"""Public evidence excerpt from an emitted Inductor wrapper.

The machine-local wrapper metadata and executable setup were removed. The AOT ID
and Graph fragment comment below are retained for structural verification.
"""

# AOT ID: ['0_inference']
# Graph fragment:
#   %arg0_1 : Tensor "f32[1048576][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sin : Tensor "f32[1048576][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg0_1,), kwargs = {})
#   %mul : Tensor "f32[1048576][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 0.5), kwargs = {})
#   %add : Tensor "f32[1048576][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %cos : Tensor "f32[1048576][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%add,), kwargs = {})
#   return %cos
