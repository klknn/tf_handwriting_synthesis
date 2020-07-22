import tensorflow.compat.v1 as tf

from tfsq import model


# tf.disable_v2_behavior()
tf.disable_eager_execution()


class ModelTest(tf.test.TestCase):
    """Tests for model."""

    def test_shape(self):
        with self.session():
            # static only
            x = tf.zeros((2, 3))
            self.assertEqual(model.shape(x), [2, 3])
            # static + dynamic
            x = tf.placeholder(tf.float32, (None, 2))
            s = model.shape(x)
            self.assertEqual(s[1], 2)
            self.assertEqual(s[0].eval({x: [[0, 0]]}), 1)


if __name__ == "__main__":
    tf.test.main()
