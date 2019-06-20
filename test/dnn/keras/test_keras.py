from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib

from test.testbase import TestBase

class TestKeras(TestBase):
    def test_keras(self):
        device_lib.list_local_devices()

        num_cores = 4
        num_CPU = 1
        num_GPU = 1

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                                device_count={'CPU': num_CPU, 'GPU': num_GPU})
        session = tf.Session(config=config)
        K.set_session(session)

        # Creates a graph.
        with tf.device('/device:GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)
        # Creates a session with log_device_placement set to True.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # Runs the op.
        print(sess.run(c))