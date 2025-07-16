import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _initialize_nmf


class BINGONMF(NMF):
    def __init__(
        self,
        n_components: int,
        alpha_h=0.1,
        step_size_w=1e-3,
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
        self.alpha_h      = alpha_h
        self.step_size_w  = step_size_w
        self.step_size_h  = step_size_h
        self.custom_max_iter = max_iter

    def _fit(self, X, *_):
        # 1. Input checks
        if X.min() < 0:
            raise ValueError(
                "Values of X should be positive."
            )

        # 2. Initialization (NNDSVD)
        W, H = _initialize_nmf(
            X,
            n_components=self.n_components,
            init=self.init,  # 'nndsvd'
            random_state=self.random_state,
        )

        prev_err = None
        for n in tqdm(range(self.custom_max_iter)):
            # 3a. Update H with PGD (incl. L1 gradient)
            grad_H = W.T.dot(W.dot(H) - X) + self.alpha_h * np.sign(H)
            H -= self.step_size_h * grad_H
            H[H < 0] = 0

            # 3b. Update W with PGD
            grad_W = (W.dot(H) - X).dot(H.T)
            W -= self.step_size_w * grad_W
            W[W < 0] = 0

            # 3c. Convergence check
            recon = W.dot(H)
            err = np.linalg.norm(X - recon, 'fro')
            if prev_err is not None and abs(prev_err - err) < self.tol:
                break
            prev_err = err

        # 4. Store results
        self.components_ = H
        return W

    def fit_transform(self, X, y=None, **fit_params):
        W = self._fit(X, **fit_params)
        return W
