import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess


class SimData:
    def __init__(self, sample_size=100, n_features=2, eps_distri="normal", **kwargs):
        self.sample_size = sample_size
        self.n_features = n_features
        self.kwargs = kwargs
        self.eps_distri = eps_distri

        self.x = self.gen_x()
        self.epsilon = self.gen_epsilon()
        self.y_signal = self.gen_y_signal()
        self.y = self.gen_y()

    def gen_x(self):
        return np.random.normal(
            scale=self.kwargs.get("x_scale", 1),
            size=(self.sample_size, self.n_features),
        )

    def gen_epsilon(self):
        if self.eps_distri == "normal":
            return np.random.normal(
                scale=self.kwargs.get("eps_scale", 1), size=(self.sample_size, 1)
            )
        elif self.eps_distri == "ar":
            ar_params = self.kwargs.get("ar_params", [1, -0.5])
            ma_params = self.kwargs.get("ma_params", None)
            ar_model = ArmaProcess(ar_params, ma_params)
            return ar_model.generate_sample(nsample=[self.sample_size, 1], burnin=200)
        elif self.eps_distri == "heteroscedasticity":
            return np.random.normal(
                scale=np.abs(np.random.normal(size=(self.sample_size, 1)))
            )
        elif self.eps_distri == "two-way-clustering":
            n1, n2 = self.kwargs.get("n1", 10), self.kwargs.get("n2", 10)
            f1 = np.random.normal(size=(n1, 1))
            f2 = np.random.normal(size=(n2, 1))
            return 1 / np.sqrt(2) * (f1 + f2.T).reshape([self.sample_size, 1])
        else:
            return np.zeros([self.sample_size, 1])

    def gen_y_signal(self):
        y_signal_null = 1.0 + self.x @ np.ones([self.n_features, 1])
        hypothesis = self.kwargs.get("hypothesis", "null")
        if hypothesis == "null":
            return y_signal_null
        elif hypothesis == "alternative":
            # See Horowitz and Spokoiny (2001) ECTA
            model_err = (
                np.exp(-0.5 * (self.x / self.kwargs["tau"]) ** 2)
                / (np.sqrt(2 * np.pi))
                * (5 / self.kwargs["tau"])
                * self.kwargs.get("scale_alternative", 1)
            )
            return y_signal_null + model_err

    def gen_y(self):
        return self.y_signal + self.epsilon

    def __repr__(self):
        return f"SimData(sample_size={self.sample_size}, n_features={self.n_features})"


class UStats:
    def kernel_test(self, u, X, N, nfeatures, const=1.06, exp=4, device="cpu"):
        K = np.ones([N, N])
        h_prod = 1.0

        for i in range(nfeatures):
            h = const * (N ** (-1 / (exp + nfeatures))) * X[:, i].std()
            X_diff = X[:, i].reshape([N, 1]) - X[:, i]
            K *= Epanechnikov(X_diff / h) / h
            h_prod *= h

        if device == "cpu":
            diag_K = np.diag(K)
            K -= np.diag(diag_K)
            I = u.T @ K @ u / N / (N - 1)
            sigma_hat = np.sqrt(
                2 * (h_prod) / N / (N - 1) * ((u.T**2) @ (K**2) @ (u**2))
            )
            T = N * (np.sqrt(h_prod)) * I / sigma_hat
        return T.squeeze(), sigma_hat, h_prod


def Epanechnikov(z: np.array) -> np.array:
    """Generate Epanechnikov Kernel evaluation at z

    Args:
        z (np.array): The locations at which to evaluate the Epanechnikov function

    Returns:
        np.array: Return E(z)
    """
    K = ((1 - z**2).clip(0)) * 0.75
    return K
