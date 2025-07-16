import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _initialize_nmf


class BINGONMF(NMF):
    def __init__(
        self,
        n_components: int,
        alpha=1e-1,
        spH=0.1,
        step_size_h=1e-3,
        max_iter=200,
        tol=1e-4,
        **kwargs
    ):
        super().__init__(
            n_components=n_components,
            init='nndsvd',
            solver='mu',  # Will be ignored
            max_iter=1,  # Overruled by the PGD solver
            tol=tol,
            **kwargs
        )
        self.alpha = alpha
        self.spH = spH
        self.step_size_h  = step_size_h
        self.custom_max_iter = max_iter

    def _fit(self, X, *_):
        # Input checks
        if X.min() < 0:
            raise ValueError(
                "Values of X should be positive."
            )

        # Initialization (NNDSVD)
        W, H = _initialize_nmf(
            X,
            n_components=self.n_components,
            init=self.init,  # 'nndsvd'
            random_state=self.random_state,
        )

        # Define function required for the Projected Gradient Descent
        def sparseness(H):
            n = H.shape[0]  # By definition, the first dimension of H is the number of channels
            L1 = np.sum(np.abs(H))
            L2 = np.sqrt(np.sum(H ** 2))
            return (np.sqrt(n) - (L1 / L2)) / (np.sqrt(n) - 1)

        def J1(H, spH):
            return abs(sparseness(H) - spH)

        # The new objective function incorporates a sparseness penalty
        def sparse_constrain(X, W, H, alpha, spH):
            norm_squared = np.linalg.norm(X - np.dot(W, H)) ** 2
            return norm_squared + alpha * J1(H, spH)

        def project_row_to_sparseness(row, target_sparseness):
            # Project a row to be non-negative, with unit L2 norm and target sparseness
            row = np.maximum(row, 0)
            if np.linalg.norm(row) == 0:
                return row
            row /= np.linalg.norm(row)  # L2 normalization

            # Adjust L1 norm to match target sparseness
            n = len(row)
            desired_L1 = (np.sqrt(n) - target_sparseness * (np.sqrt(n) - 1)) * np.linalg.norm(row)
            scale = desired_L1 / np.sum(row)
            return row * scale

        def project_H(H, target_sparseness):
            return np.array([project_row_to_sparseness(row, target_sparseness) for row in H])

        def pgd_step(X, W, H, mu, spH):
            # Gradient descent update for H
            H_new = H - mu * np.dot(W.T, (np.dot(W, H) - X))

            # Projection step
            H_new = project_H(H_new, spH)

            # Multiplicative update for W
            numerator = np.dot(X, H_new.T)
            denominator = np.dot(np.dot(W, H_new), H_new.T) + 1e-10  # avoid division by zero
            W_new = W * (numerator / denominator)

            return W_new, H_new

        prev_err = None
        for n in tqdm(range(self.custom_max_iter)):
            # Input: Wk and Hk
            # Output: Wk+1 and Hk+1

            # 1 Set: H := H - µ W.T (W H - V)

            # 2 Project each row of H to be non-negative, have unit L2 norm, and L1 norm set to achieve desired
            # sparseness

            # 3 W := W * (V H.T) / (W H H.T)

            # Sparseness is calculated as:
            # f(W, H) = ||X - WH||2,F + alpha J1(H),
            # where F is the L2 norm, alpha is an empirical value, J1(H) = |sparseness(H) - spH),
            # in which spH is an adjustable parameter between 0-1, sparseness(H) = (sqrt(n) - L1 / L2) / (sqrt(n) - 1),
            # where L1 = summation(Hi)n,i=1 and L2 = sqrt(summation((Hi)^2)n,i=1),
            # in which n is the number of detection channels

            W, H = pgd_step(X, W, H, self.step_size_h, self.spH)

            # Calculate error
            err = sparse_constrain(X, W, H, self.alpha, self.spH)
            if prev_err is not None and abs(err - prev_err) < self.tol:
                break
            prev_err = err

        self.components_ = H
        return W

    def fit_transform(self, X, y=None, **fit_params):
        W = self._fit(X, **fit_params)
        return W
