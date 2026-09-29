#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, List, Optional, Tuple

import tensorflow as tf

from .optimizer import Optimizer
from ..mappings import Mapping, MappingNetwork
from ..halt import Halt, HaltStatus


class _SSESOAPLayerState:
    """
    Per-layer state for the SS-eSOAP optimizer.

    Holds the Kronecker preconditioner matrices L and R, their eigenvector
    matrices QL and QR, the *current* eigenvalue estimates DL and DR, the Adam
    moments (stored directly in the rotated eigenbasis, in 2-D form, so that no
    reshape/rotation is needed at every step), and the two cached secant terms.

    Large rank-4 tensors use the balanced view ``[d0*d2, d1*d3]``.  This pairs
    the two spatial/Fourier axes with their corresponding channel axes and
    avoids a very large, skinny left factor.  Small tensors retain the
    conventional ``[prod(shape[:-1]), shape[-1]]`` view, for which changing
    the factorization costs more cold-start time than it saves.

    The secant terms are stored *rotated* as well:
        prev_s = Q_L^T (W_{t-1} - W_{t-2}) Q_R
        prev_g = Q_L^T G_{t-1} Q_R
    Because the update itself is produced in the eigenbasis, prev_s is simply
    -lr * U_rot and needs no matrix product; and because the Frobenius inner
    product is invariant under orthogonal rotations,
        Tr(Y^T S) = <Q_L^T Y Q_R, Q_L^T S Q_R>,
    so the whole self-scaling factor tau reduces to element-wise reductions.
    This is what makes tau essentially free (Appendix B.3 / F of the paper).

    For rank-1 parameters (biases) only the Adam moments are allocated and the
    method degenerates to plain Adam.

    Args:
        shape: Static shape of the parameter tensor (Python list of ints).
        dtype: TensorFlow dtype for all state tensors.
        eps:   Ridge used to initialise L and R (L_0 = eps*I, R_0 = eps*I).
    """

    def __init__(self, shape: list, dtype, eps: float = 1e-8):
        self.shape = [int(s) for s in shape]
        self.dtype = dtype

        # Rank-4 kernels dominate both state memory and eigensolver time.  Keras
        # convolution kernels use [spatial_0, spatial_1, in, out], while the FNO
        # weights use [in, out, mode_0, mode_1].  In both cases pairing axes
        # (0, 2) and (1, 3) gives a much better-balanced, still exact 2-D view.
        # E.g. [4, 4, 48, 48] becomes [192, 192], not [768, 48].
        legacy_m = 1
        for s in self.shape[:-1]:
            legacy_m *= s
        legacy_n = self.shape[-1]
        use_balanced_rank4 = len(self.shape) == 4 and max(legacy_m, legacy_n) >= 384

        if use_balanced_rank4:
            self.permutation = [0, 2, 1, 3]
            self.inverse_permutation = [0, 2, 1, 3]
            self.matrix_shape = [
                self.shape[0] * self.shape[2],
                self.shape[1] * self.shape[3],
            ]
        else:
            self.permutation = list(range(len(self.shape)))
            self.inverse_permutation = self.permutation
            self.matrix_shape = [legacy_m, legacy_n]

        self.m, self.n = self.matrix_shape
        self.uses_balanced_matrix = use_balanced_rank4

        self.is_matrix = (len(self.shape) > 1) and (self.m > 1)

        if self.is_matrix:
            mn = [self.m, self.n]
            self.L = tf.Variable(eps * tf.eye(self.m, dtype=dtype), trainable=False)
            self.R = tf.Variable(eps * tf.eye(self.n, dtype=dtype), trainable=False)
            self.QL = tf.Variable(tf.eye(self.m, dtype=dtype), trainable=False)
            self.QR = tf.Variable(tf.eye(self.n, dtype=dtype), trainable=False)
            # diag(Q^T L Q): exact eigenvalues right after a basis update,
            # Rayleigh quotients in the (stale) basis otherwise.
            self.DL = tf.Variable(
                tf.fill(mn[:1], tf.constant(eps, dtype)), trainable=False
            )
            self.DR = tf.Variable(
                tf.fill(mn[1:], tf.constant(eps, dtype)), trainable=False
            )
            self.exp_avg = tf.Variable(tf.zeros(mn, dtype), trainable=False)
            self.exp_avg_sq = tf.Variable(tf.zeros(mn, dtype), trainable=False)
            self.prev_s = tf.Variable(tf.zeros(mn, dtype), trainable=False)
            self.prev_g = tf.Variable(tf.zeros(mn, dtype), trainable=False)
        else:
            self.exp_avg = tf.Variable(tf.zeros(self.shape, dtype), trainable=False)
            self.exp_avg_sq = tf.Variable(tf.zeros(self.shape, dtype), trainable=False)

    def matrixize(self, tensor: tf.Tensor) -> tf.Tensor:
        """Return the optimizer's 2-D view of a parameter-shaped tensor."""
        if self.uses_balanced_matrix:
            tensor = tf.transpose(tensor, self.permutation)
        return tf.reshape(tensor, self.matrix_shape)

    def tensorize(self, matrix: tf.Tensor) -> tf.Tensor:
        """Invert ``matrixize`` without changing any tensor elements."""
        if self.uses_balanced_matrix:
            permuted_shape = [self.shape[i] for i in self.permutation]
            return tf.transpose(
                tf.reshape(matrix, permuted_shape), self.inverse_permutation
            )
        return tf.reshape(matrix, self.shape)


class OptimizerSSESOAP(Optimizer):
    """
    SS-eSOAP optimizer for IGM.

    Reference: Wang, Toftrup, Loeschcke, Wang & Anandkumar, "SS-ESOAP:
    Self-Scaled Adaptive Preconditioning for Physics-Informed Learning",
    arXiv:2608.29448 (Algorithm 2).

    SS-eSOAP is SOAP plus two mechanisms:

    1. **Self-scaling (secant-energy) correction.**  A scalar

           tau = min{1, max{tau_min, Tr(Y^T S) / Tr(S^T L^-1 S R^-1)}}

       matches the Kronecker metric to the curvature actually observed along
       the last parameter displacement S = W_{t-1} - W_{t-2}, with
       Y = G_t - G_{t-1}.  The eigenspace update is scaled by tau^{-1/2}.
       tau = 1 on most iterations; it only bites when the Kronecker metric
       over-estimates the directional curvature.

    2. **Adaptive basis trigger + variance downscaling.**  Instead of
       re-computing the eigenbasis on a fixed schedule, the relative
       off-diagonal mass of each Kronecker factor in its current basis

           rho(A, Q) = ||Q^T A Q - diag(Q^T A Q)||_F / (||Q^T A Q||_F + eps)

       is monitored; the (cubic) eigendecomposition only fires when
       rho > tau_trigger.  On a basis change the momentum is reprojected into
       the new basis while the second moment is *downscaled*, V <- gamma * V
       with gamma in {0.25, 0.5, 0.75} chosen from rho, rather than rotated
       (which is both O(m^3) and unstable under large rotations).

    Efficiency notes (this implementation):
      * For large balanced factors, the trigger statistic exploits orthogonal
        invariance and maintains diag(Q^T A Q) from the SOAP gradient rotations
        already needed by the update.  Their trigger checks therefore add no
        matrix products.
      * All optimizer state lives in the rotated basis in 2-D form, so a step
        costs exactly the two SOAP rotations (Q_L^T G Q_R and Q_L U Q_R^T)
        plus the trigger check; tau adds only element-wise reductions.
      * On a triggered update the three [m, n] states (momentum, cached secant
        displacement, cached rotated gradient) are reprojected in two batched
        matmuls.
      * Eigendecompositions are skipped entirely when the basis is still good,
        which in practice fires on ~15% of the steps.

    Only compatible with ``mapping=network``.

    Args:
        cost_fn:         Energy functional J(U, V, inputs).
        map:             Must be a MappingNetwork instance.
        halt:            Optional stopping-criterion bundle.
        lr:              Base learning rate (eta).
        beta1:           Momentum decay.
        beta2:           Second-moment / Kronecker-factor decay.
        eps:             Adam epsilon.
        tau_trigger:     Off-diagonal-mass threshold for a basis update.
        check_freq:      Iterations between two trigger checks (I_check).
        warmup:          Iterations before the trigger is armed (T_warm).
        tau_min:         Lower clip of the self-scaling factor.  tau^{-1/2}
                         multiplies the step, so tau_min bounds the step
                         amplification by tau_min^{-1/2}.
        self_scaling:    Set to False to disable the tau correction (this
                         recovers an adaptive-basis SOAP, useful for ablation).
        damping:         Ridge added before eigendecomposition, and floor on
                         the eigenvalues appearing in the tau denominator.
        weight_decay:    Decoupled weight decay (lambda), 0 by default.
        lr_drop_iter:     Optional zero-based iteration for a one-time learning
                          rate drop. Negative disables the schedule.
        lr_drop_factor:   Learning-rate multiplier after ``lr_drop_iter``.
                          Optimizer moments and preconditioners are retained.
        lr_auto_drop_patience: If positive, compare the cost over windows of
                          this many iterations and apply one drop when relative
                          improvement falls below the configured threshold.
        lr_auto_drop_warmup: First iteration used as a cost reference by the
                          automatic-drop controller.
        lr_auto_drop_rel_improvement: Minimum relative cost decrease per
                          comparison window before the automatic drop fires.
        iter_max:        Maximum number of gradient steps.
        print_cost:      Whether to display a progress bar.
        print_cost_freq: Display update frequency (iterations).
        precision:       ``'float32'`` or ``'float64'``.
        ord_grad_u:      Norm used for the velocity-gradient display metric.
        ord_grad_theta:  Norm used for the weight-gradient display metric.
        batch_size:      Number of patches per batch.
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
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        tau_trigger: float = 0.2,
        check_freq: int = 1,
        warmup: int = 0,
        tau_min: float = 0.1,
        self_scaling: bool = True,
        damping: float = 1e-8,
        weight_decay: float = 0.0,
        lr_drop_iter: int = -1,
        lr_drop_factor: float = 1.0,
        lr_auto_drop_patience: int = 0,
        lr_auto_drop_warmup: int = 500,
        lr_auto_drop_rel_improvement: float = 0.01,
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
        self.name = "ss_esoap"

        p = self.precision
        # lr and iter_max are Variables so that update_parameters() does not
        # invalidate the traced graph.
        self.lr = tf.Variable(lr, dtype=p, trainable=False)
        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)

        self.beta1 = tf.constant(beta1, dtype=p)
        self.beta2 = tf.constant(beta2, dtype=p)
        self.eps = tf.constant(eps, dtype=p)
        self.tau_trigger = tf.constant(tau_trigger, dtype=p)
        self.tau_min = tf.constant(tau_min, dtype=p)
        self.damping = tf.constant(damping, dtype=p)
        self._check_freq = max(1, int(check_freq))
        self._warmup = int(warmup)
        self._check_every_step = self._check_freq == 1 and self._warmup <= 0
        self.check_freq = tf.constant(self._check_freq, dtype=tf.int32)
        self.warmup = tf.constant(self._warmup, dtype=tf.int32)

        self.self_scaling = bool(self_scaling)
        self.weight_decay = float(weight_decay)
        self._lr_drop_iter = int(lr_drop_iter)
        self.lr_drop_factor = tf.constant(lr_drop_factor, dtype=p)
        self._lr_auto_drop_patience = int(lr_auto_drop_patience)
        self._lr_auto_drop_warmup = int(lr_auto_drop_warmup)
        self.lr_auto_drop_rel_improvement = tf.constant(
            lr_auto_drop_rel_improvement, dtype=p
        )
        if self._lr_drop_iter >= 0 and self._lr_auto_drop_patience > 0:
            raise ValueError(
                "lr_drop_iter and lr_auto_drop_patience are mutually exclusive"
            )
        drop_factor = float(lr_drop_factor)
        if not 0.0 < drop_factor <= 1.0:
            raise ValueError("lr_drop_factor must be in the interval (0, 1]")
        schedule_enabled = self._lr_drop_iter >= 0 or self._lr_auto_drop_patience > 0
        if schedule_enabled and drop_factor == 1.0:
            raise ValueError(
                "lr_drop_factor must be smaller than 1 when an LR drop is enabled"
            )
        if self._lr_auto_drop_warmup < 0:
            raise ValueError("lr_auto_drop_warmup must be >= 0")
        if float(lr_auto_drop_rel_improvement) < 0.0:
            raise ValueError("lr_auto_drop_rel_improvement must be >= 0")
        self.batch_size = int(batch_size)
        self._eps_init = float(eps)

        # gamma thresholds / values of the variance-state transition
        self._rho_hi = tf.constant(0.8, dtype=p)
        self._rho_mid = tf.constant(0.5, dtype=p)
        self._gamma = (
            tf.constant(0.25, dtype=p),
            tf.constant(0.5, dtype=p),
            tf.constant(0.75, dtype=p),
        )
        self._one = tf.constant(1.0, dtype=p)
        self._zero = tf.constant(0.0, dtype=p)

        # One tensor kernel is shared by all layers with the same matrix shape.
        # Static dimensions keep TensorFlow/cuBLAS autotuning cheap, while the
        # cache avoids inlining the full pair of conditional branches once for
        # every trainable tensor into minimize_impl's already-large graph.
        self._matrix_step_fns = {}
        # Iterations applied over the optimizer's lifetime.  The layer state
        # (moments, Kronecker factors, eigenbases) persists across minimize
        # calls, so the first-step basis shortcut and the bias corrections
        # must count from the first call, not from each call's iteration 0.
        self._global_iter = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Layer states - allocated in minimize() before the tf.function
        self._layer_states: Optional[List[_SSESOAPLayerState]] = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def update_parameters(self, iter_max: int, lr: float) -> None:
        self.iter_max.assign(iter_max)
        self.lr.assign(lr)

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Allocate per-layer SS-eSOAP state and validate the mapping before
        delegating to minimize_impl.
        """
        if not isinstance(self.map, MappingNetwork):
            raise TypeError(
                "❌ OptimizerSSESOAP requires mapping=network. "
                "For mapping=identity use lbfgs or cg_newton."
            )
        if self._layer_states is None:
            theta = self.map.get_theta()
            self._layer_states = [
                _SSESOAPLayerState(w.shape.as_list(), self.precision, self._eps_init)
                for w in theta
            ]
        return super().minimize(inputs)

    # ------------------------------------------------------------------ #
    # SS-eSOAP internals                                                 #
    # ------------------------------------------------------------------ #

    def _off_diagonal_ratio(
        self, A: tf.Tensor, diagonal: tf.Tensor
    ) -> tf.Tensor:
        """
        Relative off-diagonal mass of A from diag(Q^T A Q).

        For large factors, ``diagonal`` is maintained incrementally from the
        already-computed rotated gradient.  Orthogonal invariance gives

            ||offdiag(Q^T A Q)||_F^2 = ||A||_F^2 - ||diag(Q^T A Q)||_2^2,

        so checking the trigger needs no additional matrix product.
        """
        fro_sq = tf.reduce_sum(tf.square(A))
        off_sq = tf.maximum(
            fro_sq - tf.reduce_sum(tf.square(diagonal)), self._zero
        )
        rho = tf.sqrt(off_sq) / (tf.sqrt(fro_sq) + self.eps)
        return rho

    def _get_matrix_step_fn(self, m: int, n: int):
        """Return the one traced tensor kernel shared by all [m, n] layers."""
        key = (m, n)
        if key not in self._matrix_step_fns:
            dtype = self.precision

            def tensor_spec(shape):
                return tf.TensorSpec(shape, dtype=dtype)

            scalar = tf.TensorSpec([], dtype=dtype)
            self._matrix_step_fns[key] = tf.function(
                self._matrix_step_kernel,
                autograph=False,
                input_signature=[
                    tensor_spec([m, n]),  # G
                    tensor_spec([m, m]),  # L
                    tensor_spec([n, n]),  # R
                    tensor_spec([m, m]),  # QL
                    tensor_spec([n, n]),  # QR
                    tensor_spec([m]),  # DL
                    tensor_spec([n]),  # DR
                    tensor_spec([m, n]),  # first moment
                    tensor_spec([m, n]),  # second moment
                    tensor_spec([m, n]),  # previous secant displacement
                    tensor_spec([m, n]),  # previous rotated gradient
                    scalar,  # learning rate (passed by value; kernel is pure)
                    tf.TensorSpec([], dtype=tf.bool),
                    tf.TensorSpec([], dtype=tf.bool),
                    scalar,  # first-moment bias correction
                    scalar,  # second-moment bias correction
                ],
            )
        return self._matrix_step_fns[key]

    def _matrix_step_kernel(
        self,
        G: tf.Tensor,
        L_old: tf.Tensor,
        R_old: tf.Tensor,
        QL: tf.Tensor,
        QR: tf.Tensor,
        DL_old: tf.Tensor,
        DR_old: tf.Tensor,
        M: tf.Tensor,
        V: tf.Tensor,
        S: tf.Tensor,
        Gp: tf.Tensor,
        lr: tf.Tensor,
        should_check: tf.Tensor,
        first_step: tf.Tensor,
        bc1: tf.Tensor,
        bc2: tf.Tensor,
    ) -> Tuple[tf.Tensor, ...]:
        """
        Resource-free SS-eSOAP tensor kernel for matrix parameters.

        Keeping this as one reusable FunctionDef materially reduces the size
        and tracing cost of the outer training graph for models with many
        trainable tensors.  State persistence remains in ``_matrix_step``.
        """
        # --- 1. Kronecker factors ------------------------------------- #
        L = self.beta2 * L_old + (1.0 - self.beta2) * tf.matmul(
            G, G, transpose_b=True
        )
        R = self.beta2 * R_old + (1.0 - self.beta2) * tf.matmul(
            G, G, transpose_a=True
        )

        # The incremental path removes two cubic trigger products.  Restrict it
        # to genuinely large, two-sided factors: for tiny/skewed matrices the
        # products are cheap and the simpler direct graph has a lower cold cost.
        use_incremental_diagonal = min(G.shape[0], G.shape[1]) >= 128

        if use_incremental_diagonal:
            def _initial_basis_stats():
                # QL and QR are identities before the first update.  Reading
                # factor diagonals is exact and skips redundant identity GEMMs.
                return (
                    tf.linalg.diag_part(L),
                    tf.linalg.diag_part(R),
                    tf.zeros_like(G),
                )

            def _updated_basis_stats():
                G_left = tf.matmul(QL, G, transpose_a=True)
                G_rot_old = tf.matmul(G_left, QR)
                one_minus_beta2 = 1.0 - self.beta2
                dl = self.beta2 * DL_old + one_minus_beta2 * tf.reduce_sum(
                    tf.square(G_left), axis=1
                )
                dr = self.beta2 * DR_old + one_minus_beta2 * tf.reduce_sum(
                    tf.square(G_rot_old), axis=0
                )
                return dl, dr, G_rot_old

            DL, DR, G_rot = tf.cond(
                first_step, _initial_basis_stats, _updated_basis_stats
            )

            def _check():
                rho_l = self._off_diagonal_ratio(L, DL)
                rho_r = self._off_diagonal_ratio(R, DR)
                return tf.maximum(rho_l, rho_r)

            # Defaults check on every step.  Resolve that case in Python while
            # tracing so large repeated layers do not add a redundant branch.
            if self._check_every_step:
                rho = _check()
            else:
                rho = tf.cond(should_check, _check, lambda: -self._one)
        else:
            # Retain the original conditional structure for small factors.  It
            # has a lower first-call launch cost on the tested GPU stack.
            def _check_direct():
                LQ = tf.matmul(L, QL)
                RQ = tf.matmul(R, QR)
                dl = tf.reduce_sum(QL * LQ, axis=0)
                dr = tf.reduce_sum(QR * RQ, axis=0)
                rho_l = self._off_diagonal_ratio(L, dl)
                rho_r = self._off_diagonal_ratio(R, dr)
                return tf.maximum(rho_l, rho_r), dl, dr

            def _skip_direct():
                return -self._one, DL_old, DR_old

            rho, DL, DR = tf.cond(
                should_check, _check_direct, _skip_direct
            )

        # --- 2. Adaptive eigenbasis update ---------------------------- #
        def _rebase():
            # The explicit identity uses well-cached elementwise GPU kernels.
            # MatrixSetDiag is allocation-light but adds substantial CUDA JIT
            # latency to cold, short solves on the tested TensorFlow build.
            L_reg = L + self.damping * tf.eye(G.shape[0], dtype=self.precision)
            R_reg = R + self.damping * tf.eye(G.shape[1], dtype=self.precision)
            eL, QL_new = tf.linalg.eigh(L_reg)
            eR, QR_new = tf.linalg.eigh(R_reg)
            # basis transition matrices: X_new = (QL_new^T QL_old) X (QR_old^T QR_new)
            P_L = tf.matmul(QL_new, QL, transpose_a=True)
            P_R = tf.matmul(QR, QR_new, transpose_a=True)
            # reproject momentum + the two cached secant terms in 2 matmuls
            X = tf.stack([M, S, Gp])
            X = tf.matmul(P_L[tf.newaxis], X)
            X = tf.matmul(X, P_R[tf.newaxis])
            # variance-state transition: downscale instead of reprojecting
            gamma = tf.where(
                rho > self._rho_hi,
                self._gamma[0],
                tf.where(rho > self._rho_mid, self._gamma[1], self._gamma[2]),
            )
            return QL_new, QR_new, X[0], X[1], X[2], gamma * V, eL, eR

        def _keep():
            return QL, QR, M, S, Gp, V, DL, DR

        rebased = rho > self.tau_trigger
        QL, QR, M, S, Gp, V, DL, DR = tf.cond(
            rebased, _rebase, _keep
        )

        if use_incremental_diagonal:
            G_rot = tf.cond(
                rebased,
                lambda: tf.matmul(tf.matmul(QL, G, transpose_a=True), QR),
                lambda: tf.cond(
                    first_step,
                    lambda: tf.matmul(tf.matmul(QL, G, transpose_a=True), QR),
                    lambda: G_rot,
                ),
            )
        else:
            # Matches the original small-factor schedule: rotate once after
            # the optional basis update, in whichever basis was selected.
            G_rot = tf.matmul(tf.matmul(QL, G, transpose_a=True), QR)

        # --- 3. Eigenspace statistics --------------------------------- #
        M = self.beta1 * M + (1.0 - self.beta1) * G_rot
        V = self.beta2 * V + (1.0 - self.beta2) * tf.square(G_rot)

        # --- 4. Self-scaling correction ------------------------------- #
        if self.self_scaling:
            # c = Tr(Y^T S) and a = Tr(S^T L^-1 S R^-1), both element-wise in
            # the eigenbasis (S, Gp are already rotated; L^-1, R^-1 diagonal).
            Y_rot = G_rot - Gp
            c = tf.reduce_sum(Y_rot * S)
            d_outer = tf.maximum(DL, self.damping)[:, tf.newaxis] * tf.maximum(
                DR, self.damping
            )[tf.newaxis, :]
            a = tf.reduce_sum(tf.square(S) / d_outer)
            valid = tf.logical_and(c > self._zero, a > self._zero)
            ratio = c / tf.where(valid, a, self._one)
            tau = tf.where(
                valid,
                tf.minimum(self._one, tf.maximum(self.tau_min, ratio)),
                self._one,
            )
        else:
            tau = self._one

        # --- 5. Parameter update -------------------------------------- #
        U_rot = tf.math.rsqrt(tau) * (M / bc1) / (tf.sqrt(V / bc2) + self.eps)
        U = tf.matmul(tf.matmul(QL, U_rot), QR, transpose_b=True)

        # S_{t+1} = W_t - W_{t-1} = -lr * U, already expressed in the current
        # eigenbasis.  (With weight_decay > 0 the decay part of the
        # displacement is neglected here; it is a scalar-small correction.)
        S = -lr * U_rot

        return U, L, R, QL, QR, DL, DR, M, V, S, G_rot

    def _legacy_matrix_step(
        self,
        grad: tf.Tensor,
        state: _SSESOAPLayerState,
        should_check: tf.Tensor,
        bc1: tf.Tensor,
        bc2: tf.Tensor,
        lr: tf.Tensor,
    ) -> tf.Tensor:
        """Original SS-eSOAP path for small/non-pathological tensor shapes."""
        m, n = state.m, state.n
        G = tf.reshape(grad, [m, n])

        L = self.beta2 * state.L + (1.0 - self.beta2) * tf.matmul(
            G, G, transpose_b=True
        )
        R = self.beta2 * state.R + (1.0 - self.beta2) * tf.matmul(
            G, G, transpose_a=True
        )

        QL = state.QL.read_value()
        QR = state.QR.read_value()
        M = state.exp_avg.read_value()
        V = state.exp_avg_sq.read_value()
        S = state.prev_s.read_value()
        Gp = state.prev_g.read_value()

        def _check():
            LQ = tf.matmul(L, QL)
            RQ = tf.matmul(R, QR)
            dl = tf.reduce_sum(QL * LQ, axis=0)
            dr = tf.reduce_sum(QR * RQ, axis=0)
            rho_l = self._off_diagonal_ratio(L, dl)
            rho_r = self._off_diagonal_ratio(R, dr)
            return tf.maximum(rho_l, rho_r), dl, dr

        def _skip():
            return -self._one, state.DL.read_value(), state.DR.read_value()

        rho, DL, DR = tf.cond(should_check, _check, _skip)

        def _rebase():
            eL, QL_new = tf.linalg.eigh(
                L + self.damping * tf.eye(m, dtype=self.precision)
            )
            eR, QR_new = tf.linalg.eigh(
                R + self.damping * tf.eye(n, dtype=self.precision)
            )
            P_L = tf.matmul(QL_new, QL, transpose_a=True)
            P_R = tf.matmul(QR, QR_new, transpose_a=True)
            X = tf.stack([M, S, Gp])
            X = tf.matmul(P_L[tf.newaxis], X)
            X = tf.matmul(X, P_R[tf.newaxis])
            gamma = tf.where(
                rho > self._rho_hi,
                self._gamma[0],
                tf.where(rho > self._rho_mid, self._gamma[1], self._gamma[2]),
            )
            return QL_new, QR_new, X[0], X[1], X[2], gamma * V, eL, eR

        def _keep():
            return QL, QR, M, S, Gp, V, DL, DR

        QL, QR, M, S, Gp, V, DL, DR = tf.cond(
            rho > self.tau_trigger, _rebase, _keep
        )

        G_rot = tf.matmul(tf.matmul(QL, G, transpose_a=True), QR)
        M = self.beta1 * M + (1.0 - self.beta1) * G_rot
        V = self.beta2 * V + (1.0 - self.beta2) * tf.square(G_rot)

        if self.self_scaling:
            Y_rot = G_rot - Gp
            c = tf.reduce_sum(Y_rot * S)
            d_outer = tf.maximum(DL, self.damping)[:, tf.newaxis] * tf.maximum(
                DR, self.damping
            )[tf.newaxis, :]
            a = tf.reduce_sum(tf.square(S) / d_outer)
            valid = tf.logical_and(c > self._zero, a > self._zero)
            ratio = c / tf.where(valid, a, self._one)
            tau = tf.where(
                valid,
                tf.minimum(self._one, tf.maximum(self.tau_min, ratio)),
                self._one,
            )
        else:
            tau = self._one

        U_rot = tf.math.rsqrt(tau) * (M / bc1) / (tf.sqrt(V / bc2) + self.eps)
        U = tf.matmul(tf.matmul(QL, U_rot), QR, transpose_b=True)

        state.L.assign(L)
        state.R.assign(R)
        state.QL.assign(QL)
        state.QR.assign(QR)
        state.DL.assign(DL)
        state.DR.assign(DR)
        state.exp_avg.assign(M)
        state.exp_avg_sq.assign(V)
        state.prev_g.assign(G_rot)
        state.prev_s.assign(-lr * U_rot)

        return lr * tf.reshape(U, tf.shape(grad))

    def _matrix_step(
        self,
        grad: tf.Tensor,
        state: _SSESOAPLayerState,
        should_check: tf.Tensor,
        first_step: tf.Tensor,
        bc1: tf.Tensor,
        bc2: tf.Tensor,
        lr: tf.Tensor,
    ) -> tf.Tensor:
        """Apply the shared matrix kernel and persist one layer's state."""
        G = state.matrixize(grad)
        # Small legacy-layout tensors are cheaper to inline; a function-call
        # boundary costs more cold-start time than their tiny products save.
        # Large balanced tensors share one traced kernel per matrix shape.
        kernel = (
            self._get_matrix_step_fn(state.m, state.n)
            if state.uses_balanced_matrix
            else self._matrix_step_kernel
        )
        (
            U,
            L,
            R,
            QL,
            QR,
            DL,
            DR,
            M,
            V,
            S,
            G_rot,
        ) = kernel(
            G,
            state.L.read_value(),
            state.R.read_value(),
            state.QL.read_value(),
            state.QR.read_value(),
            state.DL.read_value(),
            state.DR.read_value(),
            state.exp_avg.read_value(),
            state.exp_avg_sq.read_value(),
            state.prev_s.read_value(),
            state.prev_g.read_value(),
            lr,
            should_check,
            first_step,
            bc1,
            bc2,
        )

        state.L.assign(L)
        state.R.assign(R)
        state.QL.assign(QL)
        state.QR.assign(QR)
        state.DL.assign(DL)
        state.DR.assign(DR)
        state.exp_avg.assign(M)
        state.exp_avg_sq.assign(V)
        state.prev_s.assign(S)
        state.prev_g.assign(G_rot)

        return lr * state.tensorize(U)

    def _adam_step(
        self,
        grad: tf.Tensor,
        state: _SSESOAPLayerState,
        bc1: tf.Tensor,
        bc2: tf.Tensor,
        lr: tf.Tensor,
    ) -> tf.Tensor:
        """Plain Adam, used for rank-1 parameters (biases)."""
        m_new = self.beta1 * state.exp_avg + (1.0 - self.beta1) * grad
        v_new = self.beta2 * state.exp_avg_sq + (1.0 - self.beta2) * tf.square(grad)
        state.exp_avg.assign(m_new)
        state.exp_avg_sq.assign(v_new)
        m_hat = m_new / bc1
        v_hat = v_new / bc2
        return lr * m_hat / (tf.sqrt(v_hat) + self.eps)

    def _ssesoap_step(
        self,
        grad: tf.Tensor,
        state: _SSESOAPLayerState,
        should_check: tf.Tensor,
        first_step: tf.Tensor,
        bc1: tf.Tensor,
        bc2: tf.Tensor,
        lr: tf.Tensor,
    ) -> tf.Tensor:
        # Static rank branch: resolved at trace time, no tf.cond overhead.
        if state.is_matrix:
            if not state.uses_balanced_matrix:
                return self._legacy_matrix_step(
                    grad, state, should_check, bc1, bc2, lr
                )
            return self._matrix_step(
                grad, state, should_check, first_step, bc1, bc2, lr
            )
        return self._adam_step(grad, state, bc1, bc2, lr)

    def _learning_rate_at(self, iteration: tf.Tensor) -> tf.Tensor:
        """Return the optional one-drop learning-rate schedule."""
        if self._lr_drop_iter < 0:
            return self.lr.read_value()
        return tf.where(
            iteration >= self._lr_drop_iter,
            self.lr * self.lr_drop_factor,
            self.lr,
        )

    def _automatic_drop_update(
        self,
        iteration: tf.Tensor,
        cost: tf.Tensor,
        active_lr: tf.Tensor,
        reference_cost: tf.Tensor,
        dropped: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Update the optional one-drop, cost-window LR controller.

        This automates only the drop time; the initial rate remains explicit.
        Its scalar state adds no objective or gradient evaluations.
        """
        if self._lr_auto_drop_patience <= 0:
            return active_lr, reference_cost, dropped, tf.constant(False)

        cost = tf.cast(cost, self.precision)
        active_lr = tf.cast(active_lr, self.precision)
        reference_cost = tf.cast(reference_cost, self.precision)
        warmup = tf.constant(self._lr_auto_drop_warmup, tf.int32)
        patience = tf.constant(self._lr_auto_drop_patience, tf.int32)
        at_reference = tf.equal(iteration, warmup)
        after_reference = tf.greater(iteration, warmup)
        at_window_end = tf.logical_and(
            after_reference,
            tf.equal(tf.math.floormod(iteration - warmup, patience), 0),
        )
        cost_is_finite = tf.math.is_finite(cost)
        reference_is_finite = tf.math.is_finite(reference_cost)
        can_compare = tf.logical_and(
            at_window_end, tf.logical_and(cost_is_finite, reference_is_finite)
        )
        relative_improvement = (reference_cost - cost) / tf.maximum(
            tf.abs(reference_cost), self.eps
        )
        should_drop = tf.logical_and(
            can_compare,
            tf.logical_and(
                tf.logical_not(dropped),
                relative_improvement < self.lr_auto_drop_rel_improvement,
            ),
        )
        active_lr = tf.where(should_drop, active_lr * self.lr_drop_factor, active_lr)
        should_update_reference = tf.logical_and(
            tf.logical_or(at_reference, at_window_end), cost_is_finite
        )
        reference_cost = tf.where(
            should_update_reference, cost, reference_cost
        )
        dropped = tf.logical_or(dropped, should_drop)
        return active_lr, reference_cost, dropped, should_drop

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs: [N, ly, lx, C] non-overlapping patches.  Each outer
                    iteration averages the gradient over all N patches
                    (batch_size B per sub-batch) and applies a single
                    SS-eSOAP update, so that the secant pair (S, Y) refers to
                    the same objective from one iteration to the next.
        """
        theta = self.map.get_theta()
        B = self.batch_size  # Python int - static at trace time
        n_batches = max(1, int(inputs.shape[0]) // B)

        # Define batch before the loops so AutoGraph infers a consistent type
        batch = inputs[0:B, :, :, :]
        batch_shape = batch.shape
        U, V = self.map.get_UV(batch)
        self._init_step_state(U, V, theta)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=self.precision, size=self.iter_max)
        active_lr = self.lr.read_value()
        lr_reference_cost = tf.constant(float("inf"), dtype=self.precision)
        lr_has_dropped = tf.constant(False)

        n_batches_f = tf.constant(float(n_batches), dtype=self.precision)

        for iter in tf.range(self.iter_max):

            lr = (
                active_lr
                if self._lr_auto_drop_patience > 0
                else self._learning_rate_at(iter)
            )

            if self._check_every_step:
                should_check = tf.constant(True)
            else:
                should_check = tf.logical_and(
                    tf.greater_equal(iter, self.warmup),
                    tf.equal(tf.math.floormod(iter, self.check_freq), 0),
                )

            step_f = tf.cast(self._global_iter + 1, self.precision)
            first_step = tf.equal(self._global_iter, 0)
            bc1 = 1.0 - tf.math.pow(self.beta1, step_f)
            bc2 = 1.0 - tf.math.pow(self.beta2, step_f)

            cost_sum = tf.constant(0.0, dtype=self.precision)
            grad_u_norm_sum = tf.constant(0.0, dtype=self.precision)
            grad_theta_norm_sum = tf.constant(0.0, dtype=self.precision)
            grad_theta_accum = [tf.zeros_like(w) for w in theta]

            # Small batch counts are faster when statically unrolled: a nested
            # while_loop otherwise serializes all gradient outputs before the
            # optimizer can launch.  Large counts use one compact loop body to
            # prevent graph construction from scaling with n_batches.
            batch_indices = (
                range(n_batches) if n_batches <= 4 else tf.range(n_batches)
            )
            for b in batch_indices:
                batch = inputs[b * B : (b + 1) * B, :, :, :]
                batch = tf.ensure_shape(batch, batch_shape)
                cost, grad_u, grad_theta = self._get_grad(batch)

                grad_theta_accum = [
                    a + g for a, g in zip(grad_theta_accum, grad_theta)
                ]

                grad_u_norm, grad_theta_norm = self._get_grad_norm(grad_u, grad_theta)
                cost_sum = cost_sum + cost
                grad_u_norm_sum = grad_u_norm_sum + grad_u_norm
                grad_theta_norm_sum = grad_theta_norm_sum + grad_theta_norm

                # Kept inside the batch loop: referencing grad_u / grad_theta at
                # the outer-loop level would make AutoGraph treat them as
                # loop-carried symbols that must be defined before tf.range.
                if self.debug_mode and b == 0:
                    self._update_debug_state(iter, cost, grad_u, grad_theta)
                    self._debug_display()

            cost_avg = cost_sum / n_batches_f
            grad_u_norm_avg = grad_u_norm_sum / n_batches_f
            grad_theta_norm_avg = grad_theta_norm_sum / n_batches_f

            # One preconditioned update per iteration, on the averaged gradient
            for w, g_accum, state in zip(theta, grad_theta_accum, self._layer_states):
                update = self._ssesoap_step(
                    g_accum / n_batches_f,
                    state,
                    should_check,
                    first_step,
                    bc1,
                    bc2,
                    lr,
                )
                if self.weight_decay > 0.0:
                    update = update + (lr * self.weight_decay) * w
                w.assign_sub(update)
            self._global_iter.assign_add(1)

            costs = costs.write(iter, cost_avg)

            if self._lr_auto_drop_patience > 0:
                (
                    active_lr,
                    lr_reference_cost,
                    lr_has_dropped,
                    lr_dropped_now,
                ) = self._automatic_drop_update(
                    iter,
                    cost_avg,
                    active_lr,
                    lr_reference_cost,
                    lr_has_dropped,
                )
                if lr_dropped_now:
                    tf.print(
                        "[ss_esoap] automatic learning-rate drop after iteration",
                        iter,
                        "new_lr=",
                        active_lr,
                    )

            # Use first batch for UV update - consistent with init above
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
