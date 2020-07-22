"""Tests for tfsq.dataset."""
import tensorflow.compat.v1 as tf

from tfsq import dataset


tf.disable_eager_execution()


class DatasetTest(tf.test.TestCase):
    """Tests for tfsq.dataset."""

    def test_extract_quoted(self):
        """Test extract_quoted."""
        self.assertEqual(
            dataset.extract_quoted('a="1"b"b""3.0"'),
            ["1", "b", "3.0"],
        )

    def test_load_tf_dataset(self):
        """Test load_tf_dataset."""
        with self.session() as sess:
            ds = dataset.load_tf_dataset("testdata")
            batch = tf.data.make_one_shot_iterator(ds).get_next()
            batch_val = sess.run(batch)
            self.assertEqual(
                batch_val["text"],
                b"A MOVE to stop Mr . Gaitskell")


if __name__ == "__main__":
    tf.test.main()
