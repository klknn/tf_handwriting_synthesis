"""Neural network model."""
from typing import Dict, List

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


def simple_rnn_body(i, ys, hs, states):
    """Simple RNN body function."""
    nh = shape(hs)[-1]
    wh = tf.get_variable(
        "wh", [nh, nh],
        initializer=tf.random_normal_initializer(stddev=0.02))
    h = tf.tanh(ys[:, i] + tf.matmul(hs[:, i], wh))
    return i + 1, ys, tf.concat((hs, h[:, tf.newaxis]), axis=1), states


def rnn(xs: tf.Tensor,
        states: Dict[str, tf.Tensor],
        scope: str,
        w_stddev: float = 0.02,
        body_fn=simple_rnn_body) -> tf.Tensor:
    """Apply recurrent neural networks.

    Args:
        xs: float tensor [batch, time, feat].
        states: dict of float tensors [batch, 1, feat]
        scope: variable scope name.
        w_stddev: weight initialization scale.
        act: activation function.

    Returns:
        float tensor [batch, time, nh].

    """
    with tf.variable_scope(scope):
        nh = shape(states["h"])[-1]
        ys = linear(xs, nh, "linear_x", w_stddev)

        def cond(i, *_):
            return i < shape(xs)[1]

        _, _, hs, states = tf.while_loop(
            cond=cond,
            body=body_fn,
            loop_vars=[0, ys, states["h"], states],
            # shape_invariants=[tf.TensorShape([]),
            #                   tf.TensorShape([None, None, nh])],
        )
        # omit the first zeros
        return hs[:, 1:], {"h": hs[:, -1]}


def lstm_body(xs: tf.Tensor, states) -> tf.Tensor:
    """Apply LSTM activation (w/o peephole connections).

    Args:
        xs: float tensor [batch, nh * 4].
        states: dict of float tensors [batch, feat]

    Returns:
        float tensor [batch, time, nh].

    """
    nb, nx = shape(xs)
    assert nx % 4 == 0
    nh = nx // 4
    gates = tf.reshape(xs, [nb, nh, 4])
    fio = tf.sigmoid(gates[:, :3])
    f = fio[:, :, 0]
    i = fio[:, :, 1]
    o = fio[:, :, 2]
    c = f * states["c"] + i * tf.tanh(gates[:, :, :, 3])
    h = o * tf.tanh(c)
    return h, c



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
        abk = tf.exp(linear(tgt, nk * 3, "linear_abk", w_stddev))
        # [nb, ntgt, 1, nk]
        b = abk[:, :, tf.newaxis, nk:2*nk]
        # TODO: be careful for one step version
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

        init_state = {"h": tf.zeros_like(h_tgt[:, :1])}
        h_tgt, _state = rnn(h_tgt, init_state, scope="rnn_tgt1")
        ws, attn = graves_attn(h_src, h_tgt, 10, "attn")
        return {
            "ht": ws,
            "attn": attn,
            "num_batch": shape(batch["text_ids"])[0]
        }
