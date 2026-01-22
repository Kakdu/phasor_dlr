import numpy as np

def ukf_initialize(n, m, sigma_V, sigma_I, sigma_phiV, sigma_phiI, gamma=10):
    """
    Initialize UKF parameters and matrices.
    """
    alpha, beta, kappa = 1e-3, 2, 0

    Q = np.diag([0.1, 0.1, 0.01, 1e-5, 1e-5]) / 500
    R = np.diag([138_000**2*sigma_V**2, 900*sigma_I**2,
                 138_000**2*sigma_V**2, 900*sigma_I**2,
                 sigma_phiV**2 + sigma_phiI**2,
                 sigma_phiV**2 + sigma_phiI**2])
    
    lambda_ = alpha**2*(n+kappa) - n
    Wm = np.full(2*n+1, 0.5/(n+lambda_))
    Wc = Wm.copy()
    Wm[0] = lambda_/(n+lambda_)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    return dict(
        alpha=alpha, beta=beta, kappa=kappa, Q=Q, R=R,
        lambda_=lambda_, Wm=Wm, Wc=Wc, gamma=gamma
    )

