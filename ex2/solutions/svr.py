# ---------- epsilon_svr.py ----------
import cvxopt
import numpy as np

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# fit(), predict()
# Do not change the function signatures
# Do not change any other code
#############################

class EpsilonSVR:
    """
    ε-Support Vector Regression (dual form).

    This implementation works only with sklearn-compatible kernels.
    The kernel must be a callable with signature
        K(X, Y) -> ndarray of shape (len(X), len(Y)),
    e.g. an instance from sklearn.gaussian_process.kernels (RBF, Matern, etc.).

    Dual optimization problem
    -------------------------
    Given training data {(x_i, y_i)}_{i=1..n}, we solve for alpha, alpha* ∈ R^n:

        minimize   1/2 (alpha - alpha*)^T K (alpha - alpha*) + epsilon 1^T (alpha + alpha*) - y^T (alpha - alpha*)
        subject to 0 ≤ alpha_i ≤ C,
                   0 ≤ alpha*_i ≤ C,
                   1^T (alpha - alpha*) = 0.

    Here K is the kernel Gram matrix. The solution defines coefficients
    (alpha - alpha*) that weight support vectors in the prediction.

    Prediction
    ----------
    For a test point x,
        f(x) = Σ_i (alpha_i - alpha*_i) K(x_i, x) + b.

    Notes
    -----
    * C > 0 controls the regularization strength (penalty for large alpha, alpha*).
    * epsilon ≥ 0 defines the “epsilon-insensitive” zone around targets y_i where
      deviations incur no loss.
    * Input normalization (scaling by max norm) can be enabled for
      numerical stability.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel=None, normalize=True):
        if kernel is None:
            raise ValueError("Provide an sklearn-compatible kernel instance (callable K(X, Y)).")
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.__sk_kernel = kernel
        self.__normalize = bool(normalize)

        # Learned params
        self.__a = None             # a
        self.__a_star = None        # a*
        self.__coef = None          # (a - a*)
        self.__bias = 0.0
        self.__training_X = None    # numpy, scaled if normalize=True
        self.__norm = 1.0
        self.__support_mask = None  # boolean mask over training samples

    # ---- Sklearn-kernel bridge ----
    def _kernel(self, X1_np, X2_np):
        return self.__sk_kernel(X1_np, X2_np)

    def fit(self, X, y):
        """
        Fit ε-SVR in the dual form using quadratic programming.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input vectors.
        y : array-like of shape (n_samples,)
            Training target values.

        Task
        ----
        * Optionally normalize X for stability.
        * Build the kernel Gram matrix K(X, X).
        * Formulate the dual quadratic program in variables z = [alpha; alpha*].
        * Solve with a QP solver (e.g. cvxopt).
        * Extract alpha, alpha*, coefficients (alpha - alpha*), and identify support vectors.
        * Compute bias b from KKT conditions using near-margin samples.

        Returns
        -------
        self : EpsilonSVR
            Fitted model with dual variables and bias stored.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, _ = X.shape

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Step 1: Normalize X if required
        if self.__normalize:
            self.__norm = np.max(np.linalg.norm(X, axis=1))
            if self.__norm < 1e-10:  # Avoid division by zero
                self.__norm = 1.0
            X_scaled = X / self.__norm
        else:
            X_scaled = X
        self.__training_X = X_scaled

        # Step 2: Compute the kernel Gram matrix K
        K = self._kernel(X_scaled, X_scaled) 

        # Step 3: Set up the QP problem
        # Objective function components
        P = np.block([
            [K, -K],
            [-K, K]
        ])
        P = P + 1e-8 * np.eye(2 * n) # numerical stability

        q = np.hstack([self.epsilon * np.ones(n) - y, 
                       self.epsilon * np.ones(n) + y])
        
        # Constraints
        G = np.vstack([-np.eye(2*n), np.eye(2*n)])
        h = np.hstack([np.zeros(2*n), self.C * np.ones(2*n)])

        A = np.hstack([np.ones(n), -np.ones(n)]).reshape(1, -1)
        b = np.array([0.0])
        # Convert to cvxopt format
        P_cvx = cvxopt.matrix(P)
        q_cvx = cvxopt.matrix(q)
        G_cvx = cvxopt.matrix(G)
        h_cvx = cvxopt.matrix(h)
        A_cvx = cvxopt.matrix(A)
        b_cvx = cvxopt.matrix(b)

        # Step 4: Solve the QP problem
        solution = cvxopt.solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
        z = np.array(solution['x']).flatten()
        alpha = z[:n]
        alpha_star = z[n:]

        self.__a = alpha
        self.__a_star = alpha_star
        self.__coef = alpha - alpha_star

        # Step 5: Identify support vectors
        tol = 1e-5
        self.__support_mask = (np.abs(self.__coef) > tol)

        # Step 6: Compute bias b using KKT conditions
        b_values = []
        free_alpha = np.where((alpha > tol) & (alpha < self.C - tol))[0]
        for i in free_alpha:
            b_i = y[i] - self.epsilon - np.dot(self.__coef, K[:, i])
            b_values.append(b_i)
        
        # Free support vectors from alpha_star (0 < alpha* < C)
        free_alpha_star = np.where((alpha_star > tol) & (alpha_star < self.C - tol))[0]
        for i in free_alpha_star:
            b_i = y[i] + self.epsilon - np.dot(self.__coef, K[:, i])
            b_values.append(b_i)

        if len(b_values) > 0:
            self.__bias = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            sv_idx = np.where(self.__support_mask)[0]
            if len(sv_idx) > 0:
                b_vals = []
                for i in sv_idx:
                    pred_i = np.dot(self.__coef, K[:, i])
                    b_vals.append(y[i] - pred_i)
                self.__bias = np.mean(b_vals)
            else:
                self.__bias = 0.0   
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        
        return self

    def _k_train_test(self, Xtest_scaled):
        return self._kernel(self.__training_X, Xtest_scaled)  # (n_train, n_test)

    def predict(self, X):
        """
        Predict regression outputs for new data using the dual form.

        For each test point x, compute:
            f(x) = Σ_i (alpha_i - alpha*_i) K(x_i, x) + b,
        where the sum runs over support vectors.

        Parameters
        ----------
        X : array-like of shape (m, n_features)
            Test input vectors.

        Returns
        -------
        y_pred : np.ndarray of shape (m,)
            Predicted target values.
        """

        if self.__coef is None:
            raise RuntimeError("Model is not fit yet.")

        X = np.asarray(X, dtype=np.float64)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        if self.__normalize:
            X_scaled = X / self.__norm
        else:
            X_scaled = X
        
        K_test = self._k_train_test(X_scaled)
        y_pred = np.dot(self.__coef, K_test) + self.__bias
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return y_pred
    

