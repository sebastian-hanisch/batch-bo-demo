"""Kern der Demo: Gauß-Prozess (Matérn-5/2, ARD), erwartete Verbesserung und vier Regeln, einen Batch von q Punkten zu wählen.

Alles auf dem Einheitswürfel, Minimierung, rauschfrei, nur numpy/scipy. Eine Einheit = eine Regel (DEMO-PLAYBOOK §1b): `matern52`, `GaussianProcess.fit/predict/sample`,
`expected_improvement`, `candidate_set`, `select_batch` (topq, kb, cl, ts), `batch_search`. Zufall nur über übergebene Generatoren.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

LS_BOUNDS = (0.05, 10.0)


def matern52(A, B, lengthscales, variance):
    """Matérn-5/2-Kern k(a, b) = σ² (1 + √5 r + 5r²/3) exp(−√5 r), r = Abstand mit Längenmaßen je Eingabe."""
    r2 = np.maximum((((A[:, None, :] - B[None, :, :]) / lengthscales) ** 2).sum(-1), 0.0)
    s = np.sqrt(5.0) * np.sqrt(r2)
    return variance * (1.0 + s + s * s / 3.0) * np.exp(-s)


class GaussianProcess:
    """Gauß-Prozess-Regression, Ausgabe standardisiert, Rauschvarianz fest klein (rauschfreie Funktionen), je Eingabe ein Längenmaß.
    Die Hyperparameter maximieren die Randwahrscheinlichkeit (L-BFGS-B)."""

    def __init__(self, start_ls=0.4, noise=1e-6):
        self.start_ls = start_ls
        self.noise = noise
        self.theta_ = None

    def _neg_log_marginal(self, theta, X, z):
        d = X.shape[1]
        K = matern52(X, X, np.exp(theta[:d]), np.exp(theta[d])) + (self.noise + 1e-8) * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, z)
        return 0.5 * z @ a + np.log(np.diag(c[0])).sum() + 0.5 * len(z) * np.log(2 * np.pi)

    def fit(self, X, y, warm=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        d = X.shape[1]
        self.y_mean, self.y_std = float(y.mean()), max(float(y.std()), 1e-9)
        z = (y - self.y_mean) / self.y_std
        bounds = [(np.log(LS_BOUNDS[0]), np.log(LS_BOUNDS[1]))] * d + [(np.log(0.05), np.log(10.0))]
        th0 = np.concatenate([np.full(d, np.log(self.start_ls)), [0.0]]) if warm is None else np.asarray(warm, float)
        res = minimize(self._neg_log_marginal, th0, args=(X, z), method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
        self.theta_ = res.x
        self.lengthscales, self.variance = np.exp(res.x[:d]), float(np.exp(res.x[d]))
        return self.condition(X, y)

    def condition(self, X, y):
        """Beobachtungen setzen (auch Scheinbeobachtungen); Hyperparameter und Standardisierung bleiben."""
        self.X = np.asarray(X, float)
        z = (np.asarray(y, float) - self.y_mean) / self.y_std
        K = matern52(self.X, self.X, self.lengthscales, self.variance) + (self.noise + 1e-8) * np.eye(len(self.X))
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, z)
        return self

    def copy_with(self, X, y):
        g = GaussianProcess(self.start_ls, self.noise)
        g.y_mean, g.y_std = self.y_mean, self.y_std
        g.lengthscales, g.variance, g.theta_ = self.lengthscales, self.variance, self.theta_
        return g.condition(X, y)

    def predict(self, Xs):
        """Mittelwert und Standardabweichung der Funktion."""
        Ks = matern52(np.asarray(Xs, float), self.X, self.lengthscales, self.variance)
        mu = Ks @ self._alpha
        v = cho_solve(self._chol, Ks.T)
        var = np.maximum(self.variance - (Ks * v.T).sum(1), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std

    def sample(self, Xs, rng, n=1):
        """n gemeinsame Stichproben der Funktion an den Stellen Xs (Feld n x len(Xs))."""
        Xs = np.asarray(Xs, float)
        Ks = matern52(Xs, self.X, self.lengthscales, self.variance)
        mu = Ks @ self._alpha
        v = cho_solve(self._chol, Ks.T)
        cov = matern52(Xs, Xs, self.lengthscales, self.variance) - Ks @ v
        cov = 0.5 * (cov + cov.T) + 1e-8 * self.variance * np.eye(len(Xs))
        L = np.linalg.cholesky(cov)
        return (mu[None, :] + (L @ rng.standard_normal((len(Xs), n))).T) * self.y_std + self.y_mean


def expected_improvement(mu, sd, best):
    """Erwartete Verbesserung (Minimierung): E[max(best − f, 0)], f ~ N(mu, sd²)."""
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def candidate_set(X, y, rng, n_random=1500, n_local=450, local_sd=0.05):
    """Kandidaten der Akquisitionsmaximierung: zufällige Punkte im Würfel plus Störungen um die drei besten Beobachtungen."""
    d = X.shape[1]
    cand = rng.random((n_random, d))
    top = X[np.argsort(y)[: min(3, len(y))]]
    per = n_local // max(len(top), 1)
    local = np.clip(np.repeat(top, per, 0) + rng.normal(0.0, local_sd, (len(top) * per, d)), 0.0, 1.0)
    return np.vstack([cand, local])


def select_batch(gp, X, y, q, rng, method):
    """Wählt q Punkte nach einer der Regeln: 'topq' (q beste EI-Punkte, ohne Abstimmung), 'kb' (Kriging Believer), 'cl' (Constant Liar mit dem besten Wert),
    'ts' (Thompson-Batch). Gibt ein Feld q x d zurück; das Modell `gp` ist auf den echten Beobachtungen angepasst."""
    best = float(np.min(y))
    cand = candidate_set(X, y, rng)
    if method == "topq":
        mu, sd = gp.predict(cand)
        ei = expected_improvement(mu, sd, best)
        return cand[np.argsort(-ei)[:q]]
    if method == "ts":
        sub = cand[rng.choice(len(cand), size=min(600, len(cand)), replace=False)]
        S = gp.sample(sub, rng, n=q)
        return sub[np.argmin(S, axis=1)]
    chosen = []
    Xa, ya, g = X, y, gp
    for _ in range(q):
        mu, sd = g.predict(cand)
        ei = expected_improvement(mu, sd, float(np.min(ya)))
        x = cand[int(np.argmax(ei))]
        chosen.append(x)
        fake = float(g.predict(x[None, :])[0][0]) if method == "kb" else best
        Xa, ya = np.vstack([Xa, x]), np.append(ya, fake)
        g = gp.copy_with(Xa, ya)
    return np.array(chosen)


def batch_search(f, d, n_start, total, q, rng, method, X0=None):
    """Bayessche Optimierung mit Batches der Größe q (q = 1 ist sequentiell, `method` dann ohne Wirkung).
    `X0` ist ein vorgegebenes Startdesign (gleich für alle Varianten eines Seeds). Gibt (X, y, Zahl der Runden, Streuung je Runde) zurück; X in Auswertungsreihenfolge, die letzte Runde darf kleiner sein.
    Streuung = mittlerer paarweiser Abstand der Punkte einer Runde (durch √d geteilt), leer bei q = 1."""
    X = rng.random((n_start, d)) if X0 is None else np.asarray(X0, float)
    y = np.array([f(x) for x in X])
    rounds, spread, theta = 0, [], None
    while len(y) < total:
        gp = GaussianProcess().fit(X, y, warm=theta)
        theta = gp.theta_
        qq = min(q, total - len(y))
        if qq == 1:
            cand = candidate_set(X, y, rng)
            mu, sd = gp.predict(cand)
            new = cand[[int(np.argmax(expected_improvement(mu, sd, float(np.min(y)))))]]
        else:
            new = select_batch(gp, X, y, qq, rng, method)
        if len(new) > 1:
            dist = np.sqrt(((new[:, None, :] - new[None, :, :]) ** 2).sum(-1))
            spread.append(float(dist[np.triu_indices(len(new), 1)].mean() / np.sqrt(d)))
        X = np.vstack([X, new])
        y = np.append(y, [f(x) for x in new])
        rounds += 1
    return X, y, rounds, spread
