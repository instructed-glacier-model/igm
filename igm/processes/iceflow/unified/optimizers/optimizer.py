#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

from .progress_optimizer import ProgressOptimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus, HaltState, StepState, DebugState
from igm.utils.math.precision import normalize_precision
from igm.utils.math.norms import compute_norm

import logging

log = logging.getLogger("werkzeug")  # for DASH POST requests...
log.setLevel(logging.ERROR)


class Optimizer(ABC):

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        debug_mode: bool = False,
        debug_freq: int = 100,
    ):
        self.name = ""
        self.cost_fn = cost_fn
        self.map = map
        self.halt = halt
        self.step_state = None
        self.halt_state = None
        self.display = ProgressOptimizer(enabled=print_cost, freq=print_cost_freq)
        self.precision = normalize_precision(precision)
        self.ord_grad_u = ord_grad_u
        self.ord_grad_theta = ord_grad_theta
        self.debug_mode = debug_mode
        self.debug_freq = debug_freq
        if self.debug_mode:
            self._init_debug_display()

    @abstractmethod
    def update_parameters(self) -> None:
        raise NotImplementedError(
            "❌ The parameters update function is not implemented in this class."
        )

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:

        if int(self.iter_max) == 0:
            return tf.zeros([0], dtype=self.precision)

        if self.halt:
            self.halt.reset_all()

        criterion_names = self.halt.criterion_names if self.halt else []
        self.display.start(int(self.iter_max), criterion_names)
        self.map.on_minimize_start(int(self.iter_max))
        costs = self.minimize_impl(inputs)
        return costs

    @abstractmethod
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The minimize_impl function is not implemented in this class."
        )

    def _get_grad_norm(
        self, grad_u: list[tf.Tensor], grad_theta: list[tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # For theta
        grad_theta_flat = self.map.flatten_theta(grad_theta)
        grad_theta_norm = compute_norm(grad_theta_flat, ord=self.ord_grad_theta)
        # For (u,v)
        grad_u_x, grad_u_y = grad_u
        # Handle case where grad_u is None (cost doesn't depend on U,V)
        if grad_u_x is None or grad_u_y is None:
            grad_u_norm = tf.constant(0.0, dtype=self.precision)
        else:
            grad_u_flat = tf.sqrt(tf.square(grad_u_x) + tf.square(grad_u_y))
            grad_u_norm = compute_norm(grad_u_flat, ord=self.ord_grad_u)
        return grad_u_norm, grad_theta_norm

    @tf.function
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        theta = self.map.get_theta()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for theta_i in theta:
                tape.watch(theta_i)
            U, V = self.map.get_UV(inputs)
            cost = self.cost_fn(U, V, inputs)

        grads = tape.gradient(cost, [U, V] + theta)
        grad_u = tuple(grads[:2])
        grad_theta = grads[2:]
        
        return cost, grad_u, grad_theta

    def _init_step_state(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        theta: Any,
    ) -> None:
        self.step_state = StepState(
            iter=tf.constant(-1, dtype=self.precision),
            u=[U, V],
            theta=theta,
            cost=tf.constant(np.nan, dtype=self.precision),
            grad_u_norm=tf.constant(np.nan, dtype=self.precision),
            grad_theta_norm=tf.constant(np.nan, dtype=self.precision),
        )

    def _update_step_state(
        self,
        iter: tf.Tensor,
        U: tf.Tensor,
        V: tf.Tensor,
        theta: Any,
        cost: tf.Tensor,
        grad_u_norm: tf.Tensor,
        grad_theta_norm: tf.Tensor,
    ) -> None:
        self.step_state = StepState(
            iter, [U, V], theta, cost, grad_u_norm, grad_theta_norm
        )

    def _check_stopping(self) -> tf.Tensor:
        """Check stopping criteria, update halt_state, and return status"""
        if self.halt is None:
            self.halt_state = HaltState.empty()
        else:
            status, values, satisfied = self.halt.check(
                self.step_state.iter, self.step_state
            )
            self.halt_state = HaltState(status, values, satisfied)

        # Check max iterations
        max_iter_reached = tf.greater_equal(self.step_state.iter + 1, self.iter_max)

        self.halt_state.status = tf.cond(
            tf.not_equal(self.halt_state.status, HaltStatus.CONTINUE.value),
            lambda: self.halt_state.status,
            lambda: tf.cond(
                max_iter_reached,
                lambda: tf.constant(HaltStatus.COMPLETED.value),
                lambda: tf.constant(HaltStatus.CONTINUE.value),
            ),
        )

        return self.halt_state.status

    def _update_display(self) -> None:
        """Update display using halt_state"""

        def update_display(iter_val, cost_val, *crit_data):
            if crit_data:
                n = len(crit_data) // 2
                values = [float(crit_data[i].numpy()) for i in range(n)]
                satisfied = [bool(crit_data[n + i].numpy()) for i in range(n)]
            else:
                values = None
                satisfied = None

            self.display.update(
                int(iter_val.numpy()),
                float(cost_val.numpy()),
                values,
                satisfied,
            )
            return 1.0

        should_update = self.display.should_update(self.step_state.iter)

        # Build args from halt_state
        py_func_args = [self.step_state.iter, self.step_state.cost]
        if self.halt_state.criterion_values and self.halt_state.criterion_satisfied:
            py_func_args.extend(self.halt_state.criterion_values)
            py_func_args.extend(self.halt_state.criterion_satisfied)

        tf.cond(
            should_update,
            lambda: tf.py_function(update_display, py_func_args, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def _finalize_display(self, halt_status: tf.Tensor) -> None:
        """Finalize display using provided halt_status"""
        if not self.display.enabled:
            return

        should_stop = tf.not_equal(halt_status, HaltStatus.CONTINUE.value)

        def finalize(status: tf.Tensor) -> int:
            status_val = int(status.numpy())
            halt_status_enum = HaltStatus(status_val)
            self.display.stop(halt_status_enum)
            return 0

        tf.cond(
            should_stop,
            lambda: tf.py_function(finalize, [halt_status], tf.int32),
            lambda: tf.constant(0),
        )

    def _update_debug_state(
        self,
        iter: tf.Tensor,
        costs: tf.Tensor,
        grad_u: list[tf.Tensor],
        grad_theta: list[tf.Tensor],
        # costs: Dict[tf.Tensor],
    ) -> None:
        self.debug_state = DebugState(iter, costs, grad_u, grad_theta)

    def _init_debug_display(self) -> None:
        """Initialize the debug figure with live updates"""
        from dash import Dash, dcc, html, Input, Output
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import threading

        # Store the latest data separately
        self._latest_grad_norm = [[0]]  # Placeholder
        self._layer_grad_norms = []  # Placeholder for layer norms
        self._latest_iter = 0

        # Create Dash app
        app = Dash(__name__)
        app.layout = html.Div(
            [
                dcc.Graph(
                    id="live-graph",
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "displaylogo": False,
                    },
                    style={"height": "90vh"},
                ),
                dcc.Interval(id="interval", interval=500, n_intervals=0),
            ]
        )

        @app.callback(Output("live-graph", "figure"), Input("interval", "n_intervals"))
        def update_graph(n):
            # Create subplots: heatmap on left, bar chart on right
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Gradient Norm Heatmap", "Layer Gradient Norms"),
                specs=[[{"type": "heatmap"}, {"type": "bar"}]],
                column_widths=[0.6, 0.4],
                horizontal_spacing=0.1,
            )

            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=self._latest_grad_norm,
                    colorscale="RdBu",
                    colorbar=dict(title="Gradient Norm", x=0.55),
                ),
                row=1,
                col=1,
            )

            # Add bar chart for layer norms
            if len(self._layer_grad_norms) > 0:
                fig.add_trace(
                    go.Bar(
                        x=[f"L{i}" for i in range(len(self._layer_grad_norms))],
                        y=self._layer_grad_norms,
                        marker=dict(color="steelblue"),
                        text=[f"{val:.2e}" for val in self._layer_grad_norms],
                        textposition="auto",
                    ),
                    row=1,
                    col=2,
                )

            # Update layout
            fig.update_layout(
                title=f"Gradient Norm Visualization (iter {self._latest_iter})",
                width=1600,
                height=900,
                uirevision="constant",
                showlegend=False,
            )

            # Update axes
            fig.update_xaxes(title_text="X", row=1, col=1, fixedrange=False)
            fig.update_yaxes(title_text="Y", row=1, col=1, fixedrange=False)
            fig.update_xaxes(title_text="Layer", row=1, col=2)
            fig.update_yaxes(title_text="Gradient Norm", type="log", row=1, col=2)

            return fig

        # Start server in background thread
        def run_server():
            app.run(debug=False, port=8050, host="127.0.0.1")

        threading.Thread(target=run_server, daemon=True).start()

    def _debug_display(self) -> None:
        """Update the debug figure. Will extend this to arbitrary cost terms (for inverse and more physically motivated problems)"""

        def _spatial_grads(*args) -> int:
            grad_u_vector = tf.reduce_mean(tf.abs(args[0]), axis=[1, 2])
            grad_u_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_u_vector), axis=0))

            grad_u_norm_np = grad_u_norm.numpy()
            self._latest_grad_norm = grad_u_norm_np.tolist()

            return 0

        def _layer_grad_norms(*grad_theta: list[tf.Tensor]) -> int:
            layer_norms = []
            for layer in grad_theta:
                layer_norm = tf.linalg.norm(tf.reshape(layer, [-1]), ord=2)
                layer_norms.append(layer_norm.numpy())
            self._layer_grad_norms = layer_norms
            return 0

        def _save_iter(iter: tf.Tensor) -> int:
            self._latest_iter = int(iter.numpy())
            return 0

        tf.py_function(_spatial_grads, [self.debug_state.grad_u], tf.int32)
        tf.py_function(_save_iter, [self.debug_state.iter], tf.int32)
        tf.py_function(_layer_grad_norms, self.debug_state.grad_theta, tf.int32)
