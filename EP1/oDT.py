import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Node:
    def __init__(self, w=None, th_star=None, left=None, right=None, label=None):
        self.w = w
        self.th_star = th_star
        self.left = left
        self.right = right
        self.label = label


class oDT:
    def __init__(self, max_depth=10, min_samples_leaf=8, n_random=5, n_thresholds=60):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_random = n_random
        self.n_thresholds = n_thresholds

    def fit(self, X, Y):
        self.root = self.build_tree(X, Y, depth=0)

    def build_tree(self, X, Y, depth=0):
        if len(np.unique(Y)) == 1:
            return Node(label=Y[0])

        if depth >= self.max_depth:
            return Node(label=self.calculate_leaf_label(Y))

        if len(Y) < 2 * self.min_samples_leaf:
            return Node(label=self.calculate_leaf_label(Y))

        w_star, th_star = self.get_best_split(X, Y)

        if w_star is None:
            return Node(label=self.calculate_leaf_label(Y))

        z = X @ w_star

        left_mask = z <= th_star
        right_mask = z > th_star

        X_left = X[left_mask]
        Y_left = Y[left_mask]
        X_right = X[right_mask]
        Y_right = Y[right_mask]

        if len(Y_left) < self.min_samples_leaf or len(Y_right) < self.min_samples_leaf:
            return Node(label=self.calculate_leaf_label(Y))

        left_subtree = self.build_tree(X_left, Y_left, depth + 1)
        right_subtree = self.build_tree(X_right, Y_right, depth + 1)

        return Node(
            w=w_star,
            th_star=th_star,
            left=left_subtree,
            right=right_subtree
        )

    def random_direction(self, m):
        w = np.random.randn(m)
        norm = np.linalg.norm(w)

        if norm == 0:
            return None

        return w / norm

    def svm_directions(self, X, Y):
        directions = []

        try:
            svm = LinearSVC(
                C=1.0,
                max_iter=5000,
                dual=False,
                tol=1e-3,
                class_weight="balanced"
            )

            svm.fit(X, Y)

            for w in svm.coef_:
                norm = np.linalg.norm(w)

                if norm > 0:
                    directions.append(w / norm)

        except Exception:
            pass

        return directions

    def lda_directions(self, X, Y):
        directions = []

        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, Y)

            for w in lda.coef_:
                norm = np.linalg.norm(w)

                if norm > 0:
                    directions.append(w / norm)

        except Exception:
            pass

        return directions

    def generate_directions(self, X, Y):
        directions = []

        directions.extend(self.svm_directions(X, Y))
        directions.extend(self.lda_directions(X, Y))

        for _ in range(self.n_random):
            w_random = self.random_direction(X.shape[1])

            if w_random is not None:
                directions.append(w_random)

        return directions

    def entropy_from_counts(self, counts):
        counts = np.asarray(counts, dtype=float)

        if counts.ndim == 1:
            total = counts.sum()

            if total == 0:
                return 0.0

            p = counts[counts > 0] / total
            return -np.sum(p * np.log2(p))

        totals = counts.sum(axis=1, keepdims=True)
        p = np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0)

        mask = p > 0
        logp = np.zeros_like(p)
        logp[mask] = np.log2(p[mask])

        return -np.sum(p * logp, axis=1)

    def best_threshold_for_projection(self, z, y_encoded, n_classes, parent_entropy):
        n = len(y_encoded)
        min_leaf = self.min_samples_leaf

        order = np.argsort(z)
        z_sorted = z[order]
        y_sorted = y_encoded[order]

        positions = np.arange(min_leaf, n - min_leaf + 1)

        if len(positions) == 0:
            return None, None

        valid = z_sorted[positions - 1] < z_sorted[positions]
        positions = positions[valid]

        if len(positions) == 0:
            return None, None

        if self.n_thresholds is not None and len(positions) > self.n_thresholds:
            idxs = np.linspace(0, len(positions) - 1, self.n_thresholds)
            idxs = np.unique(np.round(idxs).astype(int))
            positions = positions[idxs]

        one_hot = np.eye(n_classes, dtype=int)[y_sorted]
        cumulative_counts = np.cumsum(one_hot, axis=0)

        total_counts = cumulative_counts[-1]

        left_counts = cumulative_counts[positions - 1]
        right_counts = total_counts - left_counts

        left_entropy = self.entropy_from_counts(left_counts)
        right_entropy = self.entropy_from_counts(right_counts)

        n_left = positions
        n_right = n - positions

        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        gains = parent_entropy - child_entropy

        best_idx = np.argmax(gains)
        best_pos = positions[best_idx]

        best_gain = gains[best_idx]
        best_threshold = (z_sorted[best_pos - 1] + z_sorted[best_pos]) / 2

        return best_gain, best_threshold

    def get_best_split(self, X, Y):
        directions = self.generate_directions(X, Y)

        if len(directions) == 0:
            return None, None

        w_star = None
        th_star = None
        best_gain = -float("inf")

        _, y_encoded = np.unique(Y, return_inverse=True)
        n_classes = len(np.unique(y_encoded))

        parent_counts = np.bincount(y_encoded, minlength=n_classes)
        parent_entropy = self.entropy_from_counts(parent_counts)

        for w in directions:
            z = X @ w

            gain, th = self.best_threshold_for_projection(
                z,
                y_encoded,
                n_classes,
                parent_entropy
            )

            if gain is not None and gain > best_gain:
                best_gain = gain
                w_star = w
                th_star = th

        return w_star, th_star

    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        predictions = []

        for x in X:
            predictions.append(self.make_prediction(x, self.root))

        return np.array(predictions)

    def make_prediction(self, x, tree):
        if tree.label is not None:
            return tree.label

        projection = x @ tree.w

        if projection <= tree.th_star:
            return self.make_prediction(x, tree.left)

        return self.make_prediction(x, tree.right)


def bootstrap_sample(X, Y):
    n = X.shape[0]
    idxs = np.random.choice(n, size=n, replace=True)

    return X[idxs], Y[idxs]


class oRF:
    def __init__(self, n_trees=100, max_depth=10, min_samples_leaf=8,
                 n_random=5, n_thresholds=60):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_random = n_random
        self.n_thresholds = n_thresholds
        self.trees = []

    def fit(self, X, Y):
        self.trees = []

        for _ in range(self.n_trees):
            X_boot, Y_boot = bootstrap_sample(X, Y)

            tree = oDT(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_random=self.n_random,
                n_thresholds=self.n_thresholds
            )

            tree.fit(X_boot, Y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = []

        for tree in self.trees:
            all_predictions.append(tree.predict(X))

        all_predictions = np.array(all_predictions)

        final_predictions = []

        for sample_votes in all_predictions.T:
            values, counts = np.unique(sample_votes, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])

        return np.array(final_predictions)


def train_val_split(X, y, val_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    idxs = rng.permutation(n)
    n_val = int(n * val_size)

    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]

    return X[train_idxs], X[val_idxs], y[train_idxs], y[val_idxs]


def standardize(X_ref, *others):
    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0)

    std[std == 0] = 1.0

    X_ref_s = (X_ref - mean) / std
    others_s = [(X - mean) / std for X in others]

    return (X_ref_s, *others_s)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes = int(seconds // 60)
    seconds = seconds % 60

    return f"{minutes} min {seconds:.2f} s"


def clean_counter(values):
    return {int(k): int(v) for k, v in Counter(values).items()}


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    total_start = time.perf_counter()

    np.random.seed(42)

    data = np.load("/kaggle/input/competitions/pcs-3838-2026/data.npz")

    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test = data["X_test"]

    print_section("DADOS")
    print("X_train:", X_train_full.shape)
    print("y_train:", y_train_full.shape)
    print("X_test :", X_test.shape)
    print("Classes:", np.unique(y_train_full))
    print("Distribuição das classes:", clean_counter(y_train_full))

    orf_kwargs = dict(
        n_trees=100,
        max_depth=10,
        min_samples_leaf=8,
        n_random=5,
        n_thresholds=60
    )

    orthogonal_rf_kwargs = dict(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1
    )

    print_section("PARÂMETROS DA ORF")
    for key, value in orf_kwargs.items():
        print(f"{key}: {value}")

    print_section("PARÂMETROS DA RANDOM FOREST ORTOGONAL")
    for key, value in orthogonal_rf_kwargs.items():
        print(f"{key}: {value}")

    X_tr, X_val, y_tr, y_val = train_val_split(
        X_train_full,
        y_train_full,
        val_size=0.2,
        seed=42
    )

    X_tr_s, X_val_s = standardize(X_tr, X_val)

    print_section("VALIDAÇÃO LOCAL - ORF PROPOSTA")

    orf_model = oRF(**orf_kwargs)

    t0 = time.perf_counter()
    orf_model.fit(X_tr_s, y_tr)
    orf_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_val_hat_orf = orf_model.predict(X_val_s)
    orf_pred_time = time.perf_counter() - t0

    orf_acc = accuracy_score(y_val, y_val_hat_orf)

    print("Acurácia oRF                 :", orf_acc)
    print("Tempo de treino oRF          :", format_time(orf_train_time))
    print("Tempo de predição oRF        :", format_time(orf_pred_time))
    print("Distribuição prevista oRF    :", clean_counter(y_val_hat_orf))

    print_section("VALIDAÇÃO LOCAL - RANDOM FOREST ORTOGONAL")

    orthogonal_rf = RandomForestClassifier(**orthogonal_rf_kwargs)

    t0 = time.perf_counter()
    orthogonal_rf.fit(X_tr, y_tr)
    rf_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_val_hat_rf = orthogonal_rf.predict(X_val)
    rf_pred_time = time.perf_counter() - t0

    rf_acc = accuracy_score(y_val, y_val_hat_rf)

    print("Acurácia RF ortogonal         :", rf_acc)
    print("Tempo de treino RF ortogonal  :", format_time(rf_train_time))
    print("Tempo de predição RF ortogonal:", format_time(rf_pred_time))
    print("Distribuição prevista RF      :", clean_counter(y_val_hat_rf))

    print_section("COMPARAÇÃO LOCAL")

    comparison_df = pd.DataFrame({
        "Modelo": ["oRF proposta", "RF ortogonal"],
        "Acurácia validação": [orf_acc, rf_acc],
        "Tempo treino (s)": [orf_train_time, rf_train_time],
        "Tempo predição (s)": [orf_pred_time, rf_pred_time]
    })

    print(comparison_df)
    print()
    print("Diferença de acurácia oRF - RF:", orf_acc - rf_acc)

    if rf_train_time > 0:
        print("Razão tempo treino oRF/RF     :", orf_train_time / rf_train_time)

    if rf_pred_time > 0:
        print("Razão tempo predição oRF/RF   :", orf_pred_time / rf_pred_time)

    comparison_df.to_csv("comparison_orf_vs_rf.csv", index=False)

    print_section("TREINAMENTO FINAL DA ORF PARA SUBMISSÃO")

    X_train_full_s, X_test_s = standardize(X_train_full, X_test)

    final_model = oRF(**orf_kwargs)

    t0 = time.perf_counter()
    final_model.fit(X_train_full_s, y_train_full)
    final_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_test_hat = final_model.predict(X_test_s)
    final_pred_time = time.perf_counter() - t0

    submission_df = pd.DataFrame({
        "ID": np.arange(1, X_test.shape[0] + 1),
        "Prediction": np.asarray(y_test_hat).astype(int)
    })

    submission_df.to_csv("submission.csv", index=False)

    total_time = time.perf_counter() - total_start

    print("Tempo treino final oRF  :", format_time(final_train_time))
    print("Tempo predição teste oRF:", format_time(final_pred_time))
    print("Predições geradas       :", len(y_test_hat))
    print("Distribuição no teste   :", clean_counter(y_test_hat))

    print_section("SUBMISSÃO")
    print("Arquivo gerado: submission.csv")
    print("Linhas        :", len(submission_df))
    print("Colunas       :", list(submission_df.columns))
    print("Tempo total   :", format_time(total_time))
    print()
    print(submission_df.head())

    print_section("ARQUIVO AUXILIAR")
    print("comparison_orf_vs_rf.csv")


if __name__ == "__main__":
    main()