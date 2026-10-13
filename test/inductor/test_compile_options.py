# Owner(s): ["module: inductor"]
import torch
import torch.fx as fx
from torch._inductor.compile_fx import compile_fx
from torch._inductor.test_case import run_tests, TestCase


class TestCompileOptions(TestCase):
    def test_compile_fx_backward_records_original_output_strides_before_inner_compile(
        self,
    ):
        # The conv2d stride leak in #197514 only shows up on backends whose
        # fake-prop already rewrites convolution_backward to channels-last
        # (CUDA; not upstream ROCm). Guard the recording order itself so a
        # regression is caught even when the stride mismatch does not reproduce.
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        gm = fx.symbolic_trace(M())
        x = torch.randn(4, requires_grad=True)
        bw_recorded = []

        def inner_compile(gm_, *args, **kwargs):
            if kwargs.get("is_backward"):
                output = gm_.graph.find_nodes(op="output")[0]
                bw_recorded.append("original_output_strides" in output.meta)
            return gm_

        compiled = compile_fx(gm, [x], inner_compile=inner_compile)
        compiled(x).sum().backward()
        self.assertEqual(bw_recorded, [True])


if __name__ == "__main__":
    run_tests()
