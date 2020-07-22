#!python3
"""Training script."""
import itertools

import tensorflow.compat.v1 as tf

from tfsq import dataset
from tfsq import model


tf.disable_v2_behavior()

tf.flags.DEFINE_float("lr", 0.01, "learning rate.")
tf.flags.DEFINE_integer("max_tgt_len", 1024, "max target length.")
tf.flags.DEFINE_integer("max_src_len", 64, "max source length.")
tf.flags.DEFINE_integer("log_interval", 10, "logging interval.")
tf.flags.DEFINE_integer("num_epochs", 10, "epochs.")
tf.flags.DEFINE_integer("hidden_size", 128, "hidden layer size.")
tf.flags.DEFINE_integer("batch_size", 12, "parallel sequence per step.")
tf.flags.DEFINE_integer("shuffle_buffer", 1024, "parallel sequence per step.")
tf.flags.DEFINE_string("root", "data", "root data dir.")
tf.flags.DEFINE_bool("download", True, "download data or not.")
tf.flags.DEFINE_string("http_user", "", "username to download data.")
tf.flags.DEFINE_string("http_password", "", "password to download data.")


FLAGS = tf.flags.FLAGS


def setup_dataset():
    if FLAGS.download:
        dataset.download_tgz(FLAGS.http_user, FLAGS.http_password, FLAGS.root)
    ds = dataset.load_tf_dataset(root=FLAGS.root)
    ds = ds.filter(lambda x: (
        x["strokes_length"] <= FLAGS.max_tgt_len
        and x["text_length"] <= FLAGS.max_src_len))
    ds = ds.cache()
    ds = ds.shuffle(FLAGS.shuffle_buffer)
    ds = ds.padded_batch(FLAGS.batch_size)
    return ds


def train(batch, scope="model"):
    """Create a traning graph."""
    ret = model.net(
        batch=batch,
        n_vocab=dataset.STATS["num_vocab"],
        n_hidden=FLAGS.hidden_size,
        scope=scope)
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

    # TODO: define proper loss
    ret["loss"] = tf.reduce_mean(ret["ht"])
    ret["train_op"] = opt.minimize(ret["loss"], var_list=vs)
    return ret


def main(argv):
    """Run program."""
    del argv
    ds = setup_dataset()
    with tf.Session() as sess:
        batch = tf.data.make_one_shot_iterator(ds).get_next()
        tf.logging.info(f"model input: {batch}")
        ret = train(batch)
        tf.logging.info(f"model output: {ret}")

        tf.global_variables_initializer().run()

        # TODO: setup tensorboard
        for epoch in range(FLAGS.num_epochs):
            tf.logging.info(f"epoch: {epoch}")
            num_examples = dataset.STATS["num_examples"]
            num_processed = 0
            for i in itertools.count(1):
                try:
                    result = sess.run(ret)
                    num_processed += result["num_batch"]
                    prog = num_processed / num_examples * 100
                    loss = result["loss"]
                    if i % FLAGS.log_interval == 0 or prog == 100:
                        tf.logging.debug(
                            f"epoch: {epoch}, "
                            f"prog: {num_processed:,}/{num_examples:,} = "
                            f"{prog:.3f}%, "
                            f"loss: {loss:.3f}")
                except tf.errors.OutOfRangeError:
                    break
            # TODO: save checkpoints


if __name__ == "__main__":
    tf.app.run(main)
