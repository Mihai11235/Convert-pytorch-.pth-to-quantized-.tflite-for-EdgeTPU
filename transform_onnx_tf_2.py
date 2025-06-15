import onnx

onnx_model_path = "smallunet_model_test.onnx"
onnx_model = onnx.load(onnx_model_path)

from onnx_tf.backend import prepare

tf_rep = prepare(onnx_model)

tf_rep.export_graph("smallunet_model_tf_test")
