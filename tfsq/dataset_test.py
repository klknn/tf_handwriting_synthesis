"""Tests for tfsq.dataset."""
import matplotlib
import tensorflow.compat.v1 as tf

from tfsq import dataset

# disable tk
matplotlib.use("Agg")
tf.disable_eager_execution()


class DatasetTest(tf.test.TestCase):
    """Tests for tfsq.dataset."""

    def setUp(self):
        """Set up fixtures."""
        self.dataset = list(dataset.load_dataset("testdata"))

    def test_extract_quoted(self):
        """Test extract_quoted."""
        self.assertEqual(
            dataset.extract_quoted('a="1"b"b""3.0"'), ["1", "b", "3.0"],
        )

    def test_load_dataset(self):
        """Test load_tf_dataset."""
        ds = self.dataset
        self.assertEqual(len(ds), 5)
        self.assertEqual(ds[0].text, "A MOVE to stop Mr . Gaitskell")

    def test_tensor_example(self):
        """Test TensorExample."""
        tx = dataset.TensorExample.from_raw(self.dataset[0])
        self.assertEqual(tx.strokes_length, 568)

    def test_calc_stats(self):
        """Test calc_stats."""
        stats = dataset.calc_stats(self.dataset)
        self.assertEqual(stats["num_examples"], 5)

    def test_plot_strokes(self):
        """Test plot_strokes."""
        fig = dataset.plot_strokes(self.dataset[0].strokes)
        self.assertTrue(fig)

    def test_load_tf_dataset(self):
        """Test load_tf_dataset."""
        with self.session() as sess:
            ds = dataset.load_tf_dataset("testdata")
            batch = tf.data.make_one_shot_iterator(ds).get_next()
            b = sess.run(batch)
            self.assertEqual(b["text"], b"A MOVE to stop Mr . Gaitskell")


if __name__ == "__main__":
    tf.test.main()
