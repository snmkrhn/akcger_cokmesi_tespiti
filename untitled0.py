import tensorflow as tf
print("GPU var mı?", len(tf.config.list_physical_devices('GPU')) > 0)
