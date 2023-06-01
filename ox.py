# import tensorflow as tf
import tensorflow as tf
# Ustawienie wartości CUDA_VISIBLE_DEVICES
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# tf.config.list_physical_devices('GPU')

print(tf.__version__)
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print()
print()
print()
print(tf.config.list_physical_devices('CPU'))
print()
print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Sprawdzenie, czy TensorFlow korzysta z CUDA
tf.debugging.set_log_device_placement(True)

# Przykładowy kod testowy
x = tf.random.normal([1000, 1000])
y = tf.random.normal([1000, 1000])
z = tf.matmul(x, y)

print(z)