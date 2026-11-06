"""Generate engine/saqr_engine/runtime/probe_conv.onnx (tiny Conv+Relu graph used to verify providers)."""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

w = np.zeros((4, 3, 3, 3), np.float32)
for o in range(4):
    w[o] = (o + 1) * 0.1
b = np.full((4,), 0.5, np.float32)
g = helper.make_graph(
    [helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
     helper.make_node("Relu", ["c"], ["y"])],
    "saqr_probe",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 16, 16])],
    [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], producer_name="saqr")
m.ir_version = 8
onnx.checker.check_model(m)
out = Path(__file__).resolve().parents[1] / "engine/saqr_engine/runtime/probe_conv.onnx"
onnx.save(m, out)
print(out, out.stat().st_size)
