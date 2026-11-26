#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, List, Optional

from .optimizer import Optimizer
from ..mappings import Mapping, MappingNetwork
from ..halt import Halt, HaltStatus


class _SOAPLayerState:
    """
    Per-layer state for the SOAP optimizer.

    Holds the Kronecker preconditioner matrices L and R, their eigenvectors
    QL and QR, and the Adam first and second moment accumulators (stored in
    the original parameter space; rotation is applied on the fly).

    For rank-1 parameters (biases) only the Adam moments are allocated;
    L, R, QL, QR are not created.

    Args:
        shape: Static shape of the parameter tensor (Python list of ints).
        dtype: TensorFlow dtype for all state tensors.
    """

    def __init__(self, shape: list, dtype):
        self.shape = shape
        self.dtype = dtype

        # Flatten to 2D: rows = product of all dims except last, cols = last dim
        self.n = shape[-1]
        self.m = 1
        for s in shape[:-1]:
            self.m *= s

        self.is_matrix = (len(shape) > 1) and (self.m > 1)

        if self.is_matrix:
            self.L = tf.Variable(tf.eye(self.m, dtype=dtype) * 1e-8, trainable=False)
            self.R = tf.Variable(tf.eye(self.n, dtype=dtype) * 1e-8, trainable=False)
            self.QL = tf.Variable(tf.eye(self.m, dtype=dtype), trainable=False)
            self.QR = tf.Variable(tf.eye(self.n, dtype=dtype), trainable=False)

        self.exp_avg = tf.Variable(tf.zeros(shape, dtype=dtype), trainable=False)
        self.exp_avg_sq = tf.Variable(tf.zeros(shape, dtype=dtype), trainable=False)


class OptimizerSOAP(Optimizer):
    """
    SOAP optimizer for IGM.

    Maintains a Kronecker-factored preconditioner (L, R) per weight layer.
    Every `precond_freq` steps the factors are eigendecomposed to obtain
    rotation matrices QL, QR.  Adam moments are accumulated in the rotated
    eigenbasis, where the curvature is approximately diagonal, then the
    update is rotated back before being applied to the weights.

    For rank-1 parameters (biases) the method reduces to standard Adam.

    For the default IGM CNN (3×3 kernels, 32 filters) the weight matrices
    reshape to [288 × 32], so the Kronecker eigendecompositions cost
    O(288³) + O(32³) ≈ 24 M flops — negligible on GPU.

    Only compatible with ``mapping=network``.

    Args:
        cost_fn:         Energy functional J(U, V, inputs).
        map:             Must be a MappingNetwork instance.
        halt:            Optional stopping-criterion bundle.
        lr:              Adam learning rate.
        beta1:           Adam first-moment decay.
        beta2:           Adam second-moment decay (also used for L, R update).
        eps:             Adam epsilon for numerical stability.
        precond_freq:    Eigendecomposition frequency (iterations).  A value
                         of 10 gives a good accuracy/cost tradeoff.
        damping:         Regularisation added to L and R before eigendecomp.
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
        lr: float = 3e-4,
        beta1: float = 0.95,
        beta2: float = 0.999,
        eps: float = 1e-8,
        precond_freq: int = 10,
        damping: float = 1e-8,
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
        self.name = "soap"
        self.lr = tf.constant(lr, dtype=self.precision)
        self.beta1 = tf.constant(beta1, dtype=self.precision)
        self.beta2 = tf.constant(beta2, dtype=self.precision)
        self.eps = tf.constant(eps, dtype=self.precision)
        self.precond_freq = tf.constant(precond_freq, dtype=tf.int32)
        self.damping = tf.constant(damping, dtype=self.precision)
        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.batch_size = int(batch_size)

        # Layer states — allocated in minimize() before the tf.function
        self._layer_states: Optional[List[_SOAPLayerState]] = None

    def update_parameters(self, iter_max: int, lr: float) -> None:
        self.iter_max.assign(iter_max)
        self.lr = tf.constant(lr, dtype=self.precision)

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Allocate per-layer SOAP state and validate the mapping before
        delegating to minimize_impl.
        """
        if not isinstance(self.map, MappingNetwork):
            raise TypeError(
                "❌ OptimizerSOAP requires mapping=network. "
                "For mapping=identity use lbfgs or cg_newton."
            )
        if self._layer_states is None:
            theta = self.map.get_theta()
            self._layer_states = [
                _SOAPLayerState(w.shape.as_list(), self.precision) for w in theta
            ]
        return super().minimize(inputs)

    def _update_kronecker_factors(
        self, grad_2d: tf.Tensor, state: _SOAPLayerState
    ) -> None:
        """
        Update the running Kronecker factor estimates with the current gradient.

        L accumulates the row-space outer product (grad @ grad^T).
        R accumulates the column-space outer product (grad^T @ grad).
        Both are exponential moving averages with decay beta2.
        """
        state.L.assign(
            self.beta2 * state.L + (1.0 - self.beta2) * grad_2d @ tf.transpose(grad_2d)
        )
        state.R.assign(
            self.beta2 * state.R + (1.0 - self.beta2) * tf.transpose(grad_2d) @ grad_2d
        )

    def _update_eigenbasis(self, state: _SOAPLayerState) -> tf.Tensor:
        """
        Eigendecompose L and R and store the eigenvector matrices QL and QR.

        Returns a dummy scalar so this function can be used inside tf.cond.
        """
        L_reg = state.L + self.damping * tf.eye(state.m, dtype=self.precision)
        R_reg = state.R + self.damping * tf.eye(state.n, dtype=self.precision)
        _, QL = tf.linalg.eigh(L_reg)
        _, QR = tf.linalg.eigh(R_reg)
        state.QL.assign(QL)
        state.QR.assign(QR)
        return tf.constant(0)

    def _soap_step(
        self,
        grad: tf.Tensor,
        state: _SOAPLayerState,
        step: tf.Tensor,
        should_update_eigenbasis: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the SOAP update for one parameter tensor.

        For rank ≥ 2 parameters:
            1. Conditionally refresh the eigenbasis (QL, QR).
            2. Update Kronecker factors L, R with the current gradient.
            3. Rotate the gradient: grad_rot = QL^T @ grad_2d @ QR.
            4. Run Adam on grad_rot.
            5. Rotate the Adam update back and return lr * result.

        For rank-1 parameters: standard Adam update.

        Returns the update to subtract from the parameter.
        """
        # Use static rank to branch at trace time (no tf.cond overhead)
        if state.is_matrix:
            # 1. Optionally refresh eigenbasis
            tf.cond(
                should_update_eigenbasis,
                lambda: self._update_eigenbasis(state),
                lambda: tf.constant(0),
            )

            grad_2d = tf.reshape(grad, [state.m, state.n])

            # 2. Update Kronecker factors
            self._update_kronecker_factors(grad_2d, state)

            # 3. Rotate gradient into eigenbasis
            grad_rot = tf.transpose(state.QL) @ grad_2d @ state.QR  # [m, n]

            # 4. Adam in rotated basis
            m_2d = tf.reshape(state.exp_avg, [state.m, state.n])
            v_2d = tf.reshape(state.exp_avg_sq, [state.m, state.n])

            m_new = self.beta1 * m_2d + (1.0 - self.beta1) * grad_rot
            v_new = self.beta2 * v_2d + (1.0 - self.beta2) * tf.square(grad_rot)

            state.exp_avg.assign(tf.reshape(m_new, state.shape))
            state.exp_avg_sq.assign(tf.reshape(v_new, state.shape))

            step_f = tf.cast(step + 1, self.precision)
            bc1 = 1.0 - tf.math.pow(self.beta1, step_f)
            bc2 = 1.0 - tf.math.pow(self.beta2, step_f)
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            adam_rot = m_hat / (tf.sqrt(v_hat) + self.eps)  # [m, n]

            # 5. Rotate back
            update_2d = state.QL @ adam_rot @ tf.transpose(state.QR)
            return self.lr * tf.reshape(update_2d, tf.shape(grad))

        else:
            # Standard Adam for 1-D parameters
            state.exp_avg.assign(self.beta1 * state.exp_avg + (1.0 - self.beta1) * grad)
            state.exp_avg_sq.assign(
                self.beta2 * state.exp_avg_sq + (1.0 - self.beta2) * tf.square(grad)
            )
            step_f = tf.cast(step + 1, self.precision)
            bc1 = 1.0 - tf.math.pow(self.beta1, step_f)
            bc2 = 1.0 - tf.math.pow(self.beta2, step_f)
            m_hat = state.exp_avg / bc1
            v_hat = state.exp_avg_sq / bc2
            return self.lr * m_hat / (tf.sqrt(v_hat) + self.eps)
        
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
            should_update_eig = tf.equal(tf.math.floormod(iter, self.precond_freq), 0)

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
            for w, g_accum, state in zip(theta, grad_theta_accum, self._layer_states):
                w.assign_sub(self._soap_step(g_accum / n_batches_f, state, iter, should_update_eig))

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
