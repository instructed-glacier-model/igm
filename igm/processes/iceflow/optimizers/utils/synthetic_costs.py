import tensorflow as tf

def quadratic_test_cost_moderate(U, V, inputs):
    """
    Moderately conditioned quadratic test cost.
    
    Creates a spectrum of eigenvalues ranging from ~1 to ~100,
    giving a condition number around 100. This should require
    ~20 CG iterations to converge (roughly sqrt(condition number)).
    
    Uses a quadratic form with spatially varying weights.
    """
    # Target values
    U_target = tf.ones_like(U) * 2.0
    V_target = tf.ones_like(V) * 3.0
    
    # Create spatially varying weights to get a range of eigenvalues
    # This simulates having parameters with different sensitivities
    shape = tf.shape(U)
    
    # Linear ramp from 1 to 100 across spatial dimensions
    nx = tf.cast(shape[-1], tf.float32)
    x_coords = tf.range(nx, dtype=tf.float32) / (nx - 1.0)  # [0, 1]
    weights_U = 1.0 + 99.0 * x_coords  # [1, 100]
    
    # Reshape to broadcast
    weights_U = tf.reshape(weights_U, [1] * (len(shape) - 1) + [-1])
    
    # Different weighting for V (offset the spectrum slightly)
    weights_V = 2.0 + 98.0 * x_coords  # [2, 100]
    weights_V = tf.reshape(weights_V, [1] * (len(shape) - 1) + [-1])
    
    # Weighted quadratic cost
    cost_U = 0.5 * tf.reduce_sum(weights_U * tf.square(U - U_target))
    cost_V = 0.5 * tf.reduce_sum(weights_V * tf.square(V - V_target))
    
    return cost_U + cost_V

def quadratic_test_cost_hard(U, V, inputs):
    """
    Ill-conditioned quadratic with κ ≈ 1000.
    
    Solution: U = 2.0, V = 3.0 everywhere
    Final cost: 0.0
    
    This creates a very elongated "valley" that tests trust region
    radius adaptation. Eigenvalues range from 0.1 to 100.
    """
    # Target values
    U_target = tf.ones_like(U) * 2.0
    V_target = tf.ones_like(V) * 3.0
    
    # Create very ill-conditioned weighting
    shape = tf.shape(U)
    nx = tf.cast(shape[-1], tf.float32)
    
    # Exponential spacing from 0.1 to 100 (factor of 1000)
    x_coords = tf.range(nx, dtype=tf.float32) / (nx - 1.0)  # [0, 1]
    
    # log-space weighting: 10^(-1) to 10^2
    log_weights = -1.0 + 3.0 * x_coords  # [-1, 2]
    weights_U = tf.pow(10.0, log_weights)  # [0.1, 100]
    
    # Reshape to broadcast
    weights_U = tf.reshape(weights_U, [1] * (len(shape) - 1) + [-1])
    
    # Different spectrum for V (shifted)
    log_weights_V = -0.5 + 3.0 * x_coords  # [-0.5, 2.5]
    weights_V = tf.pow(10.0, log_weights_V)  # [0.316, 316]
    weights_V = tf.reshape(weights_V, [1] * (len(shape) - 1) + [-1])
    
    # Weighted quadratic cost
    cost_U = 0.5 * tf.reduce_sum(weights_U * tf.square(U - U_target))
    cost_V = 0.5 * tf.reduce_sum(weights_V * tf.square(V - V_target))
    
    return cost_U + cost_V

def quadratic_test_cost_extreme(U, V, inputs):
    """
    Extremely ill-conditioned quadratic with κ ≈ 100,000.
    
    Solution: U = 2.0, V = 3.0 everywhere
    Final cost: 0.0
    
    Eigenvalues range from 0.001 to 100.
    Expected iterations: 80-150 for trust region, 200-500+ for L-BFGS
    """
    U_target = tf.ones_like(U) * 2.0
    V_target = tf.ones_like(V) * 3.0
    
    shape = tf.shape(U)
    nx = tf.cast(shape[-1], tf.float32)
    x_coords = tf.range(nx, dtype=tf.float32) / (nx - 1.0)
    
    # log-space: 10^(-3) to 10^2 (factor of 100,000)
    log_weights = -3.0 + 5.0 * x_coords
    weights_U = tf.pow(10.0, log_weights)  # [0.001, 100]
    weights_U = tf.reshape(weights_U, [1] * (len(shape) - 1) + [-1])
    
    log_weights_V = -2.5 + 5.0 * x_coords
    weights_V = tf.pow(10.0, log_weights_V)  # [0.00316, 316]
    weights_V = tf.reshape(weights_V, [1] * (len(shape) - 1) + [-1])
    
    cost_U = 0.5 * tf.reduce_sum(weights_U * tf.square(U - U_target))
    cost_V = 0.5 * tf.reduce_sum(weights_V * tf.square(V - V_target))
    
    return cost_U + cost_V

def nonconvex_styblinski_tang(U, V, inputs):
    """
    Styblinski-Tang function (non-convex with many local minima).
    
    Global minimum: U = -2.903534, V = -2.903534 everywhere
    Global minimum value: -39.16617 * (number of elements in U + V)
    
    Highly non-convex with many local minima.
    Expected: Trust Region and Newton-CG should escape shallow local minima.
    """
    # Styblinski-Tang: f(x) = 0.5 * sum(x^4 - 16*x^2 + 5*x)
    U_term = 0.5 * tf.reduce_sum(
        tf.pow(U, 4) - 16.0 * tf.square(U) + 5.0 * U
    )
    V_term = 0.5 * tf.reduce_sum(
        tf.pow(V, 4) - 16.0 * tf.square(V) + 5.0 * V
    )
    
    return U_term + V_term

def nonconvex_sine_modulated_quadratic(U, V, inputs):
    """
    Quadratic basin with sinusoidal modulation (many shallow local minima).
    
    Global minimum: U = 2.0, V = 3.0 everywhere
    Global minimum value: 0.0
    
    Quadratic bowl with ripples. Tests robustness to noise-like features.
    """
    U_target = tf.ones_like(U) * 2.0
    V_target = tf.ones_like(V) * 3.0
    
    # Main quadratic term
    quad_U = tf.reduce_sum(tf.square(U - U_target))
    quad_V = tf.reduce_sum(tf.square(V - V_target))
    
    # Sinusoidal modulation (creates local minima)
    freq = 10.0
    amplitude = 0.1
    
    ripple_U = amplitude * tf.reduce_sum(
        tf.sin(freq * U) * tf.exp(-0.1 * tf.square(U - U_target))
    )
    ripple_V = amplitude * tf.reduce_sum(
        tf.sin(freq * V) * tf.exp(-0.1 * tf.square(V - V_target))
    )
    
    return quad_U + quad_V + ripple_U + ripple_V

def nonconvex_ackley(U, V, inputs):
    """
    Ackley function (non-convex with many local minima).
    
    Global minimum: U = 0.0, V = 0.0 everywhere
    Global minimum value: 0.0
    
    Nearly flat outer region with many local minima.
    Tests ability to navigate nearly-zero gradient regions.
    """
    a = 20.0
    b = 0.2
    c = 2.0 * 3.14159265359
    
    n_U = tf.cast(tf.size(U), tf.float32)
    n_V = tf.cast(tf.size(V), tf.float32)
    
    sum_sq_U = tf.reduce_sum(tf.square(U))
    sum_sq_V = tf.reduce_sum(tf.square(V))
    sum_cos_U = tf.reduce_sum(tf.cos(c * U))
    sum_cos_V = tf.reduce_sum(tf.cos(c * V))
    
    term1_U = -a * tf.exp(-b * tf.sqrt(sum_sq_U / n_U))
    term2_U = -tf.exp(sum_cos_U / n_U)
    
    term1_V = -a * tf.exp(-b * tf.sqrt(sum_sq_V / n_V))
    term2_V = -tf.exp(sum_cos_V / n_V)
    
    return (term1_U + term2_U + a + tf.exp(1.0) + 
            term1_V + term2_V + a + tf.exp(1.0))

def nonconvex_rastrigin(U, V, inputs):
    """
    Rastrigin function (highly multimodal, non-convex).
    
    Global minimum: U = 0.0, V = 0.0 everywhere
    Global minimum value: 0.0
    
    Has many local minima arranged in a regular lattice.
    Very challenging for gradient-based methods.
    """
    A = 10.0
    n_dims = tf.cast(tf.size(U) + tf.size(V), tf.float32)
    
    U_term = tf.reduce_sum(
        tf.square(U) - A * tf.cos(2.0 * 3.14159265359 * U)
    )
    V_term = tf.reduce_sum(
        tf.square(V) - A * tf.cos(2.0 * 3.14159265359 * V)
    )
    
    return A * n_dims + U_term + V_term