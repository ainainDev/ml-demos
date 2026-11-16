# Logistic Regression Classifier — interactive visualizer (Day 81)
#
# Fixed version of Ainain's streamlit-viz-tool.py (shared 2026-11-16).
# What was broken in the original on current scikit-learn (>= 1.7,
# the version Streamlit Cloud installs today) — verified locally on
# scikit-learn 1.9.1:
#   1. `multi_class` was REMOVED from LogisticRegression -> TypeError
#      on every run. Multiclass is now always multinomial, so the
#      control is kept visible but disabled, with an explanation.
#   2. `penalty='none'` (string) is invalid -> must be None.
#      Fixed by mapping the dropdown to the current API instead:
#      l2 -> l1_ratio=0, l1 -> l1_ratio=1, elasticnet -> l1_ratio slider,
#      none -> C=inf. Zero deprecation warnings on 1.9.1 (the old
#      `penalty=` argument is deprecated since 1.8).
#   3. `l1_ratio = int(number_input(...))` truncated 0.5 -> 0, so
#      elastic net could never mix. Now a 0.0-1.0 slider, only used
#      for elasticnet.
#   4. Invalid solver/penalty pairs (e.g. lbfgs + l1) crashed the app
#      with a raw ValueError. Now pre-checked against the Day 81
#      compatibility table with a friendly message + suggested fix.
#   5. Accuracy line said "Decision Tree" — it is Logistic Regression.
#   6. draw_meshgrid() read a global X; now takes X as an argument.
# UI labels and defaults are otherwise unchanged from his file.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Solver x penalty compatibility — Day 81 notes, section 3.
SUPPORTED = {
    "newton-cg": {"l2", "none"},
    "lbfgs": {"l2", "none"},
    "liblinear": {"l2", "l1"},
    "sag": {"l2", "none"},
    "saga": {"l2", "l1", "elasticnet", "none"},
}


def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
    else:  # Multiclass
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
    ax.scatter(X.T[0], X.T[1], c=y, cmap="rainbow")
    return X, y


def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array


plt.style.use("fivethirtyeight")

st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox("Select Dataset", ("Binary", "Multiclass"))

penalty = st.sidebar.selectbox(
    "Regularization", ("l2", "l1", "elasticnet", "none")
)

c_input = float(st.sidebar.number_input("C", value=1.0, min_value=0.01))

solver = st.sidebar.selectbox(
    "Solver", ("newton-cg", "lbfgs", "liblinear", "sag", "saga")
)

max_iter = int(st.sidebar.number_input("Max Iterations", value=100, min_value=10))

# multi_class was removed from scikit-learn 1.7; multiclass is now
# always multinomial. Kept visible so the Day 81 teaching point stays.
st.sidebar.selectbox(
    "Multi Class",
    ("auto", "ovr", "multinomial"),
    disabled=True,
    help="Removed in scikit-learn 1.7 — multiclass is now always "
    "multinomial, so this no longer changes the fit.",
)

l1_ratio = float(
    st.sidebar.slider(
        "l1 Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        disabled=(penalty != "elasticnet"),
        help="Only used with penalty='elasticnet': 0 = ridge-like, "
        "1 = lasso-like.",
    )
)

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button("Run Algorithm"):
    if penalty not in SUPPORTED[solver]:
        st.error(
            f"Solver '{solver}' does not support penalty '{penalty}'. "
            f"From the Day 81 table: {solver} supports "
            f"{', '.join(sorted(SUPPORTED[solver]))}. "
            "Try solver='saga' — it supports every penalty."
        )
    else:
        # Map the penalty dropdown onto the current scikit-learn API
        # (penalty= is deprecated since 1.8; l1_ratio / C carry it now).
        if penalty == "l2":
            clf_kwargs = {"l1_ratio": 0, "C": c_input}
        elif penalty == "l1":
            clf_kwargs = {"l1_ratio": 1, "C": c_input}
        elif penalty == "elasticnet":
            clf_kwargs = {"l1_ratio": l1_ratio, "C": c_input}
        else:  # penalty == "none": no regularization
            clf_kwargs = {"l1_ratio": 0, "C": float("inf")}

        try:
            clf = LogisticRegression(
                solver=solver, max_iter=max_iter, **clf_kwargs
            )
            clf.fit(X_train, y_train)
        except Exception as e:  # noqa: BLE001 - show it, don't crash
            st.error(f"Could not fit with these settings: {e}")
        else:
            orig.empty()
            y_pred = clf.predict(X_test)

            XX, YY, input_array = draw_meshgrid(X)
            labels = clf.predict(input_array)

            ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap="rainbow")
            plt.xlabel("Col1")
            plt.ylabel("Col2")
            orig = st.pyplot(fig)
            st.subheader(
                "Accuracy for Logistic Regression  "
                + str(round(accuracy_score(y_test, y_pred), 2))
            )
