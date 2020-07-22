"""Tests for tfsq.model."""
import numpy as np
import tensorflow.compat.v1 as tf

from tfsq import model


tf.disable_eager_execution()


class ModelTest(tf.test.TestCase):
    """Tests for tfsq.model."""

    def test_shape(self):
        """Tests for shape."""
        with self.session():
            # static only
            x = tf.zeros((2, 3))
            self.assertEqual(model.shape(x), [2, 3])
            # static + dynamic
            x = tf.placeholder(tf.float32, (None, 2))
            s = model.shape(x)
            self.assertEqual(s[1], 2)
            self.assertEqual(s[0].eval({x: [[0, 0]]}), 1)

    def test_rnns(self):
        """Tests for RNN variants."""
        with self.session() as sess:
            x = tf.placeholder(tf.float32, (None, None, 4))
            for Class in [model.RNN, model.LSTM]:
                # define
                net = Class(2, scope=Class.__name__)
                y, states = net(x, net.init_states(x))
                # and run
                tf.global_variables_initializer().run()
                sess.run([y, states], {x: np.zeros((5, 3, 4))})


if __name__ == "__main__":
    tf.test.main()
