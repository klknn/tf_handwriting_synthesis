"""Neural network model."""
from typing import Callable, Dict, List

import tensorflow.compat.v1 as tf


def shape(x: tf.Tensor) -> List[tf.Tensor]:
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def linear(xs: tf.Tensor, nh: int, scope: str,
           w_stddev: float = 0.02) -> tf.Tensor:
    """"Transform a sequence by linear weight and bias.

    Args:
        xs: float tensor [batch, time, feat].
        nh: number of output dim.
        scope: variable scope name.
        w_stddev: weight initialization scale.

    Returns:
        float tensor [batch, time, nh].

    """
    with tf.variable_scope(scope):
        nx = shape(xs)[-1]
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0))
        wx = tf.get_variable(
            "wx", [nx, nh],
            initializer=tf.random_normal_initializer(stddev=w_stddev))
        return tf.matmul(xs, wx) + b


def rnn(xs: tf.Tensor,
        nh: int,
        scope: str,
        w_stddev: float = 0.02,
        act: Callable[[tf.Tensor], tf.Tensor] = tf.tanh) -> tf.Tensor:
    """Apply recurrent neural networks."""
    with tf.variable_scope(scope):
        ys = linear(xs, nh, "linear_x", w_stddev)
        wh = tf.get_variable(
            "wh", [nh, nh],
            initializer=tf.random_normal_initializer(stddev=w_stddev))

        def cond(i, _):
            return i < shape(xs)[1]

        def body(i, hs):
            h = act(ys[:, i] + tf.matmul(hs[:, i], wh))
            h = h[:, tf.newaxis]
            return i + 1, tf.concat((hs, h), axis=1)

        _, hs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[tf.constant(0), tf.zeros_like(ys[:, 0:1])],
            # shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None, nh])],
        )
        # omit the first zeros
        return hs[:, 1:]


def graves_attn(src: tf.Tensor, tgt: tf.Tensor, nk: int, scope: str,
                w_stddev: float = 0.02) -> tf.Tensor:
    """Compute context vector and attention matrix in Eq. (46-51)

    A. Graves https://arxiv.org/pdf/1308.0850.pdf

    Args:
        src: float tensor [nb, nsrc, nh]
        tgt: float tensor [nb, ntgt, ng]
        nk: number of Gaussians in attention

    Returns:
        float tensor [nb, ntgt, nh]

    """
    with tf.variable_scope(scope):
        abk = tf.exp(linear(tgt, nk * 3, scope, w_stddev))
        # [nb, ntgt, 1, nk]
        b = abk[:, :, tf.newaxis, nk:2*nk]
        k = tf.cumsum(abk[:, :, tf.newaxis, 2*nk:], axis=1)

        nsrc = shape(src)[1]
        u = tf.reshape(tf.range(nsrc, dtype=k.dtype), (1, 1, nsrc, 1))
        # [nb, ntgt, nsrc, nk]
        e = tf.exp(-b * (k - u))
        # [nb, ntgt, nk, 1]
        a = abk[:, :, :nk, tf.newaxis]
        # [nb, ntgt, nsrc]
        attn = tf.squeeze(tf.matmul(e, a, name="mm_attn"), axis=-1)
        # [nb, ntgt, nh]
        w = tf.matmul(attn, src, name="mm_context")
        return w, attn


def net(batch: Dict[str, tf.Tensor], n_vocab: int, n_hidden: int,
        w_stddev: float = 0.02, scope="model") -> Dict[str, tf.Tensor]:
    """Top-level network definition."""
    with tf.variable_scope(scope):
        embed = tf.get_variable(
            "embed", [n_vocab, n_hidden],
            initializer=tf.random_normal_initializer(stddev=w_stddev))
        h_src = tf.gather(embed, batch["text_ids"], axis=0)  # [nb, nt, nh]

        # concat x, y, eos (omit time stamp), [batch, ntgt, 3]
        tgt = tf.concat(
            (batch["strokes"][:, :, :2], batch["end_flags"][:, :, tf.newaxis]),
            axis=-1)
        # shift 1 frame in network input
        tgt_input = tf.concat((tf.zeros_like(tgt[:, :1]), tgt[:, :-1]), axis=1)
        h_tgt = linear(tgt_input, n_hidden, "linear_tgt")
        h_tgt = rnn(h_tgt, n_hidden, "rnn_tgt1")
        ws, attn = graves_attn(h_src, h_tgt, 10, "attn")
        return {
            "ht": ws,
            "attn": attn,
            "num_batch": shape(batch["text_ids"])[0]
        }
