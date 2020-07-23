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

    def test_gaussian_mixture(self):
        """Tests for gaussian_mixture."""
        with self.session() as sess:
            x = tf.random.normal((2, 3, 4))
            y = model.gaussian_mixture(x, n_out=2, n_gauss=5, scope="gauss")
            z = tf.zeros((2, 3, 2))
            p = model.gaussian_mixture_pdf(z, **y)

            tf.global_variables_initializer().run()
            actual, actual_p = sess.run([y, p])
            self.assertEqual(actual["weight"].shape, (2, 3, 5))
            self.assertEqual(actual["mean"].shape, (2, 3, 5, 2))
            self.assertEqual(actual["cov"].shape, (2, 3, 5, 2, 2))
            # check tf.softmax on weight
            np.testing.assert_allclose(
                actual["weight"].sum(-1), np.float32(1), rtol=1e-6
            )
            diag = actual["cov"].transpose(3, 4, 0, 1, 2).diagonal()
            # check tf.exp on cov diag
            self.assertTrue(np.all(diag >= 0))
            # check tf.tanh on cov non-diag
            non_diag = actual["cov"].copy()
            for i0 in range(non_diag.shape[0]):
                for i1 in range(non_diag.shape[1]):
                    for i2 in range(non_diag.shape[2]):
                        np.fill_diagonal(non_diag[i0, i1, i2], 0)
            self.assertTrue(np.all(non_diag >= -1))
            self.assertTrue(np.all(non_diag <= 1))
            # check prob
            print(actual_p)
            self.assertTrue(np.all(actual_p <= 1))
            self.assertTrue(np.all(actual_p >= 0))


if __name__ == "__main__":
    tf.test.main()
