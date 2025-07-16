import torch
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _initialize_nmf

class BINGONMF_GPU(NMF):
    def __init__(
        self,
        n_components: int,
        alpha=1e-1,
        spH=0.1,
        step_size_h=1e-3,
        max_iter=200,
        tol=1e-4,
        device=None,
        **kwargs
    ):
        super().__init__(
            n_components=n_components,
            init='nndsvd',  # Will be ignored
            solver='mu',  # Overruled by the PGD solver
            max_iter=1,
            tol=tol,
            **kwargs
        )
        self.alpha = alpha
        self.spH = spH
        self.step_size_h = step_size_h
        self.custom_max_iter = max_iter
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define function required for the Projected Gradient Descent
    def _sparseness(self, H):
        n = H.shape[0]  # By definition, the first dimension of H is the number of channels
        L1 = torch.sum(torch.abs(H))
        L2 = torch.norm(H)
        return (torch.sqrt(torch.tensor(n, dtype=torch.float32, device=self.device)) - (L1 / L2)) / (torch.sqrt(torch.tensor(n, dtype=torch.float32, device=self.device)) - 1)

    def _J1(self, H):
        return torch.abs(self._sparseness(H) - self.spH)

    # The new objective function incorporates a sparseness penalty
    def _objective(self, X, W, H):
        frob = torch.norm(X - torch.matmul(W, H)) ** 2
        return frob + self.alpha * self._J1(H)

    def _project_row(self, row):
        # Project a row to be non-negative, with unit L2 norm and target sparseness
        row = torch.clamp(row, min=0)
        norm = torch.norm(row)
        if norm == 0:
            return row
        row = row / norm  # L2 normalization

        # Adjust L1 norm to match target sparseness
        n = row.shape[0]
        desired_L1 = (torch.sqrt(torch.tensor(n, dtype=torch.float32, device=self.device)) - self.spH * (torch.sqrt(torch.tensor(n, dtype=torch.float32, device=self.device)) - 1)) * norm
        scale = desired_L1 / torch.sum(row)
        return row * scale

    def _project_H(self, H):
        return torch.stack([self._project_row(row) for row in H])

    def _pgd_step(self, X, W, H):
        # Gradient descent update for H
        H_new = H - self.step_size_h * torch.matmul(W.T, torch.matmul(W, H) - X)

        # Projection step
        H_new = self._project_H(H_new)

        # Multiplicative update for W
        numerator = torch.matmul(X, H_new.T)
        denominator = torch.matmul(torch.matmul(W, H_new), H_new.T) + 1e-10  # avoid division by zero
        W_new = W * (numerator / denominator)

        return W_new, H_new

    def _fit(self, X, *_):
        if X.min() < 0:
            raise ValueError("Values of X should be positive.")

        # Convert input to torch tensor
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Initialization (NNDSVD)
        W_init, H_init = _initialize_nmf(
            X,
            n_components=self.n_components,
            init=self.init,
            random_state=self.random_state,
        )
        W = torch.tensor(W_init, dtype=torch.float32, device=self.device)
        H = torch.tensor(H_init, dtype=torch.float32, device=self.device)

        prev_err = None
        for _ in range(self.custom_max_iter):
            W, H = self._pgd_step(X_torch, W, H)

            # Calculate error
            err = self._objective(X_torch, W, H)
            if prev_err is not None and torch.abs(err - prev_err) < self.tol:
                break
            prev_err = err

        self.components_ = H.cpu().detach().numpy()
        return W.cpu().detach().numpy()

    def fit_transform(self, X, y=None, **fit_params):
        W = self._fit(X, **fit_params)
        return W
