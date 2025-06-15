import tensorflow as tf

# Load the incorrectly saved model
loaded = tf.saved_model.load("smallunet_model_tf_test")
concrete_func = loaded.signatures["serving_default"]

# Patch to handle NHWC input (normal TensorFlow input)
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 256, 256, 3], dtype=tf.float32)])
def patched_func(input_tensor):
    # (batch, height, width, channels) → (batch, channels, height, width)
    input_tensor = tf.transpose(input_tensor, [0, 3, 1, 2])
    return concrete_func(input_tensor)

# Save new patched model
tf.saved_model.save(loaded, "smallunet_model_tf_fixed_test", signatures={"serving_default": patched_func})