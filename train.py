#!python3
"""Training script."""

import itertools

import numpy as np
import tensorflow.compat.v1 as tf

import dataset


tf.disable_v2_behavior()

tf.flags.DEFINE_float("lr", 0.01, "learning rate.")
tf.flags.DEFINE_integer("log_interval", 10, "logging interval.")
tf.flags.DEFINE_integer("num_epochs", 10, "epochs.")
tf.flags.DEFINE_integer("hidden_size", 128, "hidden layer size.")
tf.flags.DEFINE_integer("batch_size", 128, "parallel sequence per step.")
tf.flags.DEFINE_integer("shuffle_buffer", 1024, "parallel sequence per step.")
tf.flags.DEFINE_string("root", "data", "root data dir.")

FLAGS = tf.flags.FLAGS


def _setup_dataset():
    ds = dataset.load_tf_dataset(root=FLAGS.root)
    ds = ds.cache()
    ds = ds.shuffle(FLAGS.shuffle_buffer)
    ds = ds.padded_batch(FLAGS.batch_size)
    return ds


def shape(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def rnn(xs, nh, scope, w_stddev=0.02, act=tf.tanh):
    with tf.variable_scope(scope):
        nb, nt, nx = shape(xs)
        b = tf.get_variable("b", [nh],
                            initializer=tf.constant_initializer(0))
        wx = tf.get_variable("wx", [nx, nh],
                             initializer=tf.random_normal_initializer(stddev=w_stddev))
        wh = tf.get_variable("wh", [nh, nh],
                             initializer=tf.random_normal_initializer(stddev=w_stddev))
        ys = tf.matmul(xs, wx) + b

        def body(i, hs):
            h = act(ys[:, i] + tf.matmul(hs[:, i], wh))
            h = h[:, tf.newaxis]
            return i + 1, tf.concat((hs[:, :i], h, hs[:, i+1:]), axis=1)

        _, hs = tf.while_loop(
            cond=lambda i, prev: i < nt,
            body=body,
            loop_vars=[tf.constant(0), tf.zeros_like(ys)],
        )
        # omit the first zeros
        return hs[:, 1:]


def graves_attn_step(h, kappa_prev, nsrc):
    """WIP: Eq. (46--51) A. Graves https://arxiv.org/pdf/1308.0850.pdf"""
    nh = shape(h)[1]
    nk = shape(kappa_prev)[1]
    assert nk * 3 == nh
    h = tf.exp(h)
    # [nb, nk, 1]
    a = h[:, :nk, tf.newaxis]
    # [nb, 1, nk]
    b = h[:, tf.newaxis, nk:2*nk]
    k = h[:, tf.newaxis, 2*nk:] + kappa_prev
    # [1, nsrc, 1]
    u = tf.range(nsrc, dtype=k.dtype)[tf.newaxis, :, tf.newaxis]
    # [nb, nsrc, nk]
    e = tf.exp(-b * (k - u))
    # [nb, nsrc]
    return tf.matmul(e, a)[:, :, 0]


def model(batch, scope="model"):
    with tf.variable_scope(scope):
        ret = dict()
        embed = tf.get_variable("embed", [dataset.STATS["num_vocab"], FLAGS.hidden_size],
                                initializer=tf.random_normal_initializer(stddev=0.02))
        ht = tf.gather(embed, batch["text_ids"], axis=0)  # [nb, nt, nh]
        ht = rnn(ht, FLAGS.hidden_size, "rnn_embed")
        ret["ht"] = ht
        ret["num_batch"] = shape(batch["text_ids"])[0]
        return ret


def train(batch, scope="model"):
    ret = model(batch, scope)
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    # TODO: define proper loss
    ret["loss"] = tf.reduce_mean(ret["ht"])
    ret["train_op"] = opt.minimize(ret["loss"], var_list=vs)
    return ret


def main(argv):
    del argv
    ds = _setup_dataset()
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
