"""Neural network model."""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import tensorflow.compat.v1 as tf


def shape(x: tf.Tensor) -> List[tf.Tensor]:
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def linear(
    xs: tf.Tensor, nh: int, scope: str, w_stddev: float = 0.02
) -> tf.Tensor:
    """Transform a sequence by linear weight and bias.

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
            "wx",
            [nx, nh],
            initializer=tf.random_normal_initializer(stddev=w_stddev),
        )
        return tf.matmul(xs, wx) + b


class RNN(tf.Module):
    """Basic recurrent neural network."""

    States = Dict[str, tf.Tensor]

    def __init__(self, nh: int, scope: str, w_stddev=0.02):
        """Initialize class.

        Args:
            nh: number of output dim.
            scope: variable scope name.
            w_stddev: weight initialization scale.

        """
        super().__init__(name=scope)
        self.nh = nh
        self.w_stddev = w_stddev
        self.scope = scope

    def init_states(self, xs: tf.Tensor) -> RNN.States:
        """Initialize states.

        Args:
            xs: float tensor of [batch, time, feat]

        Returns:
            states used in `recurrent`.

        """
        # why tf.zeros((1, 1, self.nh)) won't work here?
        nb, nt, _ = shape(xs[:, :1])
        return {"out": tf.zeros((nb, nt, self.nh), dtype=xs.dtype)}

    def transform_input(self, xs: tf.Tensor) -> tf.Tensor:
        """Transform input sequence.

        Args:
            xs: float tensor of [batch, time, feat]

        Returns:
            float tensor of [batch, time, new_feat]

        """
        with tf.variable_scope(self.scope):
            return linear(xs, self.nh, "linear_x", self.w_stddev)

    def recurrent(
        self, i: tf.Tensor, ys: tf.Tensor, hs: tf.Tensor, states: RNN.States
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, RNN.States]:
        """Recurrent output with states.

        Args:
            i: frame counter (int).
            ys: output of `transform_input`.
            hs: final output of `__call__`.
            states: truncated state dict.

        Returns:
            updated args for the next iteration.

        """
        with tf.variable_scope(self.scope):
            wh = tf.get_variable(
                "wh",
                [self.nh, self.nh],
                initializer=tf.random_normal_initializer(stddev=self.w_stddev),
            )
            h = tf.tanh(ys[:, tf.newaxis, i] + tf.matmul(states["out"], wh))
            return i + 1, ys, tf.concat((hs, h), axis=1), {"out": h}

    def __call__(
        self, xs: tf.Tensor, states: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Apply recurrent neural networks.

        Args:
            xs: float tensor [batch, time, feat].
            states: dict of float tensors [batch, 1, feat]
            scope: variable scope name.
            w_stddev: weight initialization scale.

        Returns:
            float tensor [batch, time, nh].

        """
        with tf.variable_scope(self.scope):
            ys = self.transform_input(xs)

            def cond(i, *_):
                return i < shape(xs)[1]

            _, _, hs, states = tf.while_loop(
                cond=cond,
                body=self.recurrent,
                loop_vars=[0, ys, states["out"], states],
            )
            # omit the first zeros
            return hs[:, 1:], states


class LSTM(RNN):
    """Long short-term memory net."""

    def init_states(self, xs: tf.Tensor) -> RNN.States:
        """Initialize states.

        Args:
            xs: float tensor of [batch, time, feat]

        Returns:
            states used in `recurrent`.

        """
        nb, nt, _ = shape(xs[:, :1])
        return {
            "out": tf.zeros((nb, nt, self.nh), dtype=xs.dtype),
            "cell": tf.zeros((nb, nt, self.nh), dtype=xs.dtype),
        }

    def transform_input(self, xs: tf.Tensor) -> tf.Tensor:
        """Transform input sequence.

        Args:
            xs: float tensor of [batch, time, feat]

        Returns:
            float tensor of [batch, time, new_feat]

        """
        with tf.variable_scope(self.scope):
            return linear(xs, self.nh * 4, "linear_x", self.w_stddev)

    def recurrent(
        self, i: tf.Tensor, ys: tf.Tensor, hs: tf.Tensor, states: RNN.States
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, RNN.States]:
        """Recurrent output with states.

        Args:
            i: frame counter (int).
            ys: output of `transform_input`.
            hs: final output of `__call__`.
            states: truncated state dict.

        Returns:
            updated args for the next iteration.

        TODO: peephole connection.

        """
        with tf.variable_scope(self.scope):
            wh = tf.get_variable(
                "wh",
                [self.nh, self.nh * 4],
                initializer=tf.random_normal_initializer(stddev=self.w_stddev),
            )
            # [nb, 1, nh * 4]
            zs = ys[:, tf.newaxis, i] + tf.matmul(states["out"], wh)
            gates = tf.reshape(zs, [-1, 1, self.nh, 4])
            fio = tf.sigmoid(gates[:, :, :, :3])
            fgate = fio[:, :, :, 0]
            igate = fio[:, :, :, 1]
            ogate = fio[:, :, :, 2]
            cell = fgate * states["cell"] + igate * tf.tanh(gates[:, :, :, 3])
            out = ogate * tf.tanh(cell)
            new_states = {"out": out, "cell": cell}
            return i + 1, ys, tf.concat((hs, out), axis=1), new_states


def graves_attn(
    src: tf.Tensor, tgt: tf.Tensor, nk: int, scope: str, w_stddev: float = 0.02
) -> tf.Tensor:
    """Compute context vector and attention matrix.

    See also: Eq. (46-51) A. Graves https://arxiv.org/pdf/1308.0850.pdf

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
        b = abk[:, :, tf.newaxis, nk : 2 * nk]
        # TODO: be careful for one step version
        k = tf.cumsum(abk[:, :, tf.newaxis, 2 * nk :], axis=1)

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


def gaussian_mixture(
    xs: tf.Tensor,
    n_out,
    n_gauss: int,
    scope: str,
    non_diag_scale: float = 0.01,
) -> Dict[str, tf.Tensor]:
    """Transform Gaussian mixture pdf.

    Eq. (18 - 25) in A. Graves https://arxiv.org/pdf/1308.0850.pdf

    Args:
        xs: float tensor of [batch, time, feat]
        n_out: number of output dim.
        n_gauss: number of Gaussians.

    Returns:
        dict of three float tensors:
        - weight [batch, time, n_gauss] (0 < w < 1, sum(w) = 1)
        - mean [batch, time, n_gauss, n_out]
        - cov [batch, time, n_gauss n_out, n_out] (0 < diag, -1 < non-diag < 1)

    """
    with tf.variable_scope(scope):
        nb, nt, _ = shape(xs)
        ys = linear(xs, n_gauss * (1 + n_out + n_out * n_out), "linear_x")
        ys = tf.reshape(ys, [nb, nt, n_gauss, -1])
        # [batch, time, n_gauss, n_out, n_out]
        c = ys[:, :, :, 1 + n_out :]
        c = tf.reshape(c, [nb, nt, n_gauss, n_out, n_out])
        diag = tf.exp(tf.linalg.diag_part(c))
        cov = tf.tanh(c)
        upper = tf.linalg.band_part(cov, 0, n_out - 1)
        non_diag = upper + tf.transpose(upper, [0, 1, 2, 4, 3])
        non_diag *= non_diag_scale
        cov = tf.linalg.set_diag(non_diag, diag)
        return {
            "weight": tf.nn.softmax(ys[:, :, :, 0]),
            "mean": ys[:, :, :, 1 : 1 + n_out],
            "cov": cov,
        }


def gaussian_mixture_pdf(
    xs: tf.Tensor, weight: tf.Tensor, mean: tf.Tensor, cov: tf.Tensor
) -> tf.Tensor:
    """Compute pdf of xs on Gaussian mixtures.

    Args:
        xs: float tensor of [batch, time, n_out]
        weight: float tensor of [batch, time, n_gauss] (0 < w < 1, sum(w) = 1)
        mean: float tensor of [batch, time, n_gauss, n_out]
        cov: float tensor of [batch, time, n_gauss n_out, n_out]
            (0 < diag, -1 < non-diag < 1)

    Returns:
        float tensor of [batch, time]

    """
    nb, nt, n_out = shape(xs)
    # [batch, time, n_gauss, 1, n_out]
    xm = (xs[:, :, tf.newaxis] - mean)[:, :, :, tf.newaxis]
    # [batch, time, n_gauss, 1, 1]
    num = tf.exp(-0.5 * xm @ cov @ tf.transpose(xm, [0, 1, 2, 4, 3]))
    # [batch, time, n_gauss]
    num = tf.squeeze(num, [-2, -1])
    den = (2 * np.pi) ** (1 / n_out) * tf.linalg.det(cov) ** 0.5
    # [batch, time]
    return tf.reduce_sum(weight * num / den, -1)


def net(
    batch: Dict[str, tf.Tensor],
    n_vocab: int,
    n_hidden: int,
    n_gauss: int,
    scope="model",
    w_stddev: float = 0.02,
    rnn_type: type = LSTM,
) -> Dict[str, tf.Tensor]:
    """Top-level network definition."""
    with tf.variable_scope(scope):
        nb = shape(batch["text_ids"])[0]
        embed = tf.get_variable(
            "embed",
            [n_vocab, n_hidden],
            initializer=tf.random_normal_initializer(stddev=w_stddev),
        )
        h_src = tf.gather(embed, batch["text_ids"], axis=0)  # [nb, nt, nh]

        # concat x, y, eos (omit time stamp), [batch, ntgt, 3]
        tgt = tf.concat(
            (batch["strokes"][:, :, :2], batch["end_flags"][:, :, tf.newaxis]),
            axis=-1,
        )
        # shift 1 frame in network input
        tgt_input = tf.concat((tf.zeros_like(tgt[:, :1]), tgt[:, :-1]), axis=1)
        h_tgt = linear(tgt_input, n_hidden, "linear_tgt")

        # forward RNNs
        rnn1 = rnn_type(n_hidden, scope="rnn1", w_stddev=w_stddev)
        h1, _s1 = rnn1(h_tgt, rnn1.init_states(h_tgt))
        # FIXME: mask by text length
        ws, attn = graves_attn(h_src, h1, 10, "attn")

        rnn2 = rnn_type(n_hidden, scope="rnn2", w_stddev=w_stddev)
        h2, _s2 = rnn2(ws, rnn2.init_states(ws))

        h_out = linear(h2, n_hidden, "linear_h_out")
        params = gaussian_mixture(
            h_out, n_out=2, n_gauss=n_gauss, scope="gauss_param"
        )
        stroke_logprob = tf.log(gaussian_mixture_pdf(tgt[:, :, :2], **params))
        stroke_loss = tf.reduce_sum(
            batch["strokes_weight"] * -stroke_logprob
        ) / tf.cast(nb, tf.float32)

        eos_pred = tf.squeeze(tf.sigmoid(linear(h_out, 1, "linear_eos")), -1)
        eos_logprob = batch["end_flags"] * tf.log(eos_pred)
        eos_logprob += (1 - batch["end_flags"]) * tf.log(1 - eos_pred)
        eos_loss = tf.reduce_sum(
            batch["strokes_weight"] * -eos_logprob
        ) / tf.cast(nb, tf.float32)

        return {
            "loss": stroke_loss + eos_loss,
            "stroke_loss": stroke_loss,
            "eos_loss": eos_loss,
            "attn": attn,  # FIXME: mask
            "num_batch": nb,
        }
