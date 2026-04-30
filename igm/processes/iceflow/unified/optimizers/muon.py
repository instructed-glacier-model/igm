#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, List, Optional

from .optimizer import Optimizer
from ..mappings import Mapping, MappingNetwork
from ..halt import Halt, HaltStatus


@tf.function(reduce_retracing=True)
def _zeropower_via_newtonschulz5(G: tf.Tensor, steps: int) -> tf.Tensor:
    """
    Compute the orthogonal polar factor of G via Newton-Schulz iteration.

    Iterates the quintic polynomial X_{k+1} = a·X + b·(XX^T)X + c·(XX^T)²X
    with coefficients chosen so each singular value converges toward 1.

    The input is transposed internally if wide (more columns than rows), then
    transposed back, so the result always matches the shape of G.

    Args:
        G:     2D tensor [m, n].
        steps: number of iterations (5 is sufficient for float32).

    Returns:
        Tensor of the same shape as G with singular values close to 1.
    """
    a = tf.constant(3.4445, dtype=G.dtype)
    b = tf.constant(-4.7750, dtype=G.dtype)
    c = tf.constant(2.0315, dtype=G.dtype)

    transposed = tf.shape(G)[0] < tf.shape(G)[1]
    G = tf.cond(transposed, lambda: tf.transpose(G), lambda: G)
    G = G / (tf.norm(G) + tf.cast(1e-7, G.dtype))

    for _ in range(steps):
        A = G @ tf.transpose(G)
        G = a * G + (b * A + c * (A @ A)) @ G

    return tf.cond(transposed, lambda: tf.transpose(G), lambda: G)


class OptimizerMuon(Optimizer):
    """
    Muon optimizer for IGM.

    Applies Nesterov momentum followed by Newton-Schulz orthogonalization to
    each weight matrix, giving a unit-spectral-norm update direction.  This
    acts as a cheap per-layer preconditioner that equalises step sizes across
    all singular directions of each weight matrix.

    Parameters with rank ≥ 2 (conv kernels, Dense kernels) receive the
    orthogonalized update scaled by `lr`.  Parameters with rank 1 (biases)
    receive plain Nesterov SGD scaled by `lr_1d`.

    Only compatible with ``mapping=network``.

    Args:
        cost_fn:         Energy functional J(U, V, inputs).
        map:             Must be a MappingNetwork instance.
        halt:            Optional stopping-criterion bundle.
        lr:              Learning rate for matrix parameters.
        momentum:        Nesterov momentum coefficient.
        ns_steps:        Newton-Schulz iterations (5 is enough for float32).
        lr_1d:           Learning rate for 1-D parameters (biases, etc.).
        iter_max:        Maximum number of gradient steps.
        print_cost:      Whether to display a progress bar.
        print_cost_freq: Display update frequency (iterations).
        precision:       ``'float32'`` or ``'float64'``.
        ord_grad_u:      Norm used for the velocity-gradient display metric.
        ord_grad_theta:  Norm used for the weight-gradient display metric.
        batch_size:  Batch size.
    """

    def __init__(
        self,
        cost_fn: Callable,
        map: Mapping,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        lr_1d: float = 3e-4,
        iter_max: int = int(1e5),
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            cost_fn,
            map,
            halt,
            print_cost,
            print_cost_freq,
            precision,
            ord_grad_u,
            ord_grad_theta,
            **kwargs,
        )
        self.name = "muon"
        self.lr = tf.constant(lr, dtype=self.precision)
        self.momentum = tf.constant(momentum, dtype=self.precision)
        self.ns_steps = ns_steps
        self.lr_1d = tf.constant(lr_1d, dtype=self.precision)
        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.batch_size = tf.constant(batch_size, dtype=tf.int32)

        # Momentum buffers — allocated in minimize() before the tf.function
        self._buf: Optional[List[tf.Variable]] = None

    def update_parameters(self, iter_max: int, lr: float) -> None:
        self.iter_max.assign(iter_max)
        self.lr = tf.constant(lr, dtype=self.precision)

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Allocate momentum buffers and validate mapping before delegating to
        minimize_impl.
        """
        if not isinstance(self.map, MappingNetwork):
            raise TypeError(
                "❌ OptimizerMuon requires mapping=network. "
                "For mapping=identity use lbfgs or cg_newton."
            )
        if self._buf is None:
            theta = self.map.get_theta()
            self._buf = [tf.Variable(tf.zeros_like(w), trainable=False) for w in theta]
        return super().minimize(inputs)

    def _muon_step(self, grad: tf.Tensor, buf: tf.Variable) -> tf.Tensor:
        """
        Compute the Muon update for one parameter tensor.

        Nesterov momentum is accumulated in `buf`, then the lookahead gradient
        is orthogonalized (rank ≥ 2) or left as-is (rank 1), and multiplied by
        the appropriate learning rate.

        Returns the update to subtract from the parameter.
        """
        # Nesterov momentum
        buf.assign(self.momentum * buf + grad)
        g = grad + self.momentum * buf

        # Use static rank (known at trace time for Keras variables)
        if len(grad.shape) > 1:
            flat = tf.reshape(g, [-1, grad.shape[-1]])
            orth = _zeropower_via_newtonschulz5(flat, self.ns_steps)
            return self.lr * tf.reshape(orth, tf.shape(grad))
        else:
            return self.lr_1d * g

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        B = self.batch_size
        n_batches = tf.maximum([1], [inputs.shape[0] // B])[0]

        # Define batch before loops so AutoGraph can infer a consistent type
        batch = inputs[0:B, :, :, :]
        batch_shape = batch.shape

        theta = self.map.get_theta()
        U, V = self.map.get_UV(batch)
        self._init_step_state(U, V, theta)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=self.precision, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):
            cost_sum = tf.constant(0.0, dtype=self.precision)
            grad_u_norm_sum = tf.constant(0.0, dtype=self.precision)
            grad_theta_norm_sum = tf.constant(0.0, dtype=self.precision)

            # Initialise accumulator with zeros matching each weight's shape
            grad_theta_accum = [tf.zeros_like(w) for w in theta]

            for b in tf.range(n_batches):
                batch = inputs[b * B : (b + 1) * B, :, :, :]
                batch = tf.ensure_shape(batch, batch_shape)

                cost, grad_u, grad_theta = self._get_grad(batch)

                grad_theta_accum = [a + g for a, g in zip(grad_theta_accum, grad_theta)]

                grad_u_norm, grad_theta_norm = self._get_grad_norm(grad_u, grad_theta)
                cost_sum += cost
                grad_u_norm_sum += grad_u_norm
                grad_theta_norm_sum += grad_theta_norm

            n_batches_f = tf.cast(n_batches, dtype=self.precision)
            cost_avg = cost_sum / n_batches_f
            grad_u_norm_avg = grad_u_norm_sum / n_batches_f
            grad_theta_norm_avg = grad_theta_norm_sum / n_batches_f

            # Apply averaged gradients once per iteration
            for w, g_accum, buf in zip(theta, grad_theta_accum, self._buf):
                w.assign_sub(self._muon_step(g_accum / n_batches_f, buf))

            costs = costs.write(iter, cost_avg)

            # Use first batch for UV update — consistent with init above
            U, V = self.map.get_UV(inputs[0:B, :, :, :])
            self._update_step_state(
                iter, U, V, theta, cost_avg, grad_u_norm_avg, grad_theta_norm_avg
            )

            halt_status = self._check_stopping()
            self._update_display()
            self.map.on_step_end(iter)
            iter_last = iter

            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)
        return costs.stack()[: iter_last + 1]
