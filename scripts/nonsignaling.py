"""PBR non-signaling test (Tabia et al.), extracted from the PBR analysis notebook (see notebooks/pbr_protocol.ipynb)."""
import itertools
from collections import Counter
from typing import Optional

import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator


class NonSignalingRegression(BaseEstimator):
    def __init__(self, players: int, questions: int, answers: int, P_x: np.ndarray | None = None):
        if players > 2:
            raise NotImplementedError("No support yet for more than 2 players")
        
        self.players = players
        self.questions = questions
        self.answers = answers

        shape = self.players*(self.questions,)

        # If not provided, assume all questions asked uniformly
        if P_x is None:
            P_x = np.ones(shape)
            P_x /= P_x.size
        else:
            if P_x.shape != shape:
                raise ValueError(f"Expected input distribution P(x) with shape {shape}, got {P_x.shape}")
            
        self.P_x = P_x
        self._kl_div = 0
        self._R: np.ndarray = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # X should be rows of the form (x, y, a, b)
        if X.ndim != 2 or X.shape[1] != 2 * self.players:
            raise ValueError(f"Expected sample array of shape (n, 2*players), got: {X.shape}")

        # Sparse structure: only store distribution for values we can actually get
        query_set = set(tuple(map(int, x)) for x in zip(*self.P_x.nonzero()))
        self._query_to_idx = {k: i for i, k in enumerate(query_set)}

        answer_set: set[tuple[int, ...]] = set(itertools.product(range(self.answers), repeat=self.players))
        self._answer_to_idx = {k: i for i, k in enumerate(answer_set)}

        # Obtain the (normalized) frequencies f(a|x)
        f_xa_dict = self._get_frequencies(X)
        
        # Convert frequencies to an array
        shape = (len(query_set), len(answer_set))
        f_xa = np.zeros(shape, dtype=float)
        for x in query_set:
            for a in answer_set:
                xa = x + a
                f_xa[self._query_to_idx[x], self._answer_to_idx[a]] = f_xa_dict.get(xa, 0)

        # Construct the convex optimization problem, here we assume all questions are equally likely
        P = cp.Variable(shape, nonneg=True)

        P_x = np.zeros((len(query_set), 1))
        nonzero_indices = self.P_x.nonzero()

        for x in zip(*nonzero_indices):
            x = tuple(map(int, x))
            P_x[self._query_to_idx[x]] = self.P_x[*x]

        kl = cp.kl_div(f_xa, P)
        rescaled_kl = cp.multiply(P_x, kl)
        objective = cp.Minimize(cp.sum(rescaled_kl))
        constraints = self._get_constraints(P)
        problem = cp.Problem(objective, constraints)

        # Solve and store the results
        self._kl_div = problem.solve(solver=cp.SCS)
        p_star = P.value
        self._R = f_xa / (p_star + 1e-9)
    
    def predict(self, X: np.ndarray):
        if X.ndim != 2 or X.shape[1] != 2 * self.players:
            raise ValueError(f"Expected sample array of shape (n, 2*players), got: {X.shape}")

        result = np.zeros(X.shape[0])

        # Given samples (x, a), return the values R(x, a)
        for i, row in enumerate(X.astype(int)):
            xa = tuple(map(int, row))
            x, a = xa[:self.players], xa[self.players:]

            # This question was never in our training data
            if (xi := self._query_to_idx.get(x, None)) is None:
                raise ValueError(f"Query {x} was not in training data")
            
            # This pair (x, a) wasn't in the training data, which can happen
            # for low probability pairs. f(a|x) = 0
            elif (ai := self._answer_to_idx.get(a, None)) is None:
                result[i] = 0

            else:
                result[i] = self._R[xi, ai]

        return result
    
    def _get_idx(self, xa: tuple[int, ...]):
        x, a = xa[:self.players], xa[self.players:]

        xi = self._query_to_idx[x]
        ai = self._answer_to_idx[a]
        return xi, ai
        
    def _get_constraints(self, P: cp.Variable):
        constraints: list[cp.Constraint] = []

        query_set = set(self._query_to_idx)
        answer_set = set(self._answer_to_idx)

        # Proper probability sum_{a} p(a|x) = 1
        for x in query_set:
            samples = []
            for a in answer_set:
                xi, ai = self._get_idx(x + a)
                samples.append(P[xi, ai])
            
            constraints.append(cp.sum(samples) == 1)
        
        # No-signaling
        for x1, x2 in itertools.combinations(query_set, 2):
            # From Bob to Alice
            if x1[0] == x2[0]:
                x = x1[0]
                y, y_prime = x1[1], x2[1]
                for a in range(self.answers):
                    samples = [P[*self._get_idx((x, y, a, b))] for b in range(self.answers)]
                    lhs = cp.sum(samples)

                    samples = [P[*self._get_idx((x, y_prime, a, b))] for b in range(self.answers)]
                    rhs = cp.sum(samples)
                    constraints.append(lhs == rhs)
            
            # From Alice to Bob
            elif x1[1] == x2[1]:
                x, x_prime = x1[0], x2[0]
                y = x1[1]
                for b in range(self.answers):
                    samples = [P[*self._get_idx((x, y, a, b))] for a in range(self.answers)]
                    lhs = cp.sum(samples)

                    samples = [P[*self._get_idx((x_prime, y, a, b))] for a in range(self.answers)]
                    rhs = cp.sum(samples)
                    constraints.append(lhs == rhs)

        return constraints

    def _get_frequencies(self, X: np.ndarray):
        raw_samples: list[list[int]] = X.astype(int).tolist()
        N_xa = Counter(map(tuple, raw_samples))

        raw_queries: list[list[int]] = X[..., :self.players].astype(int).tolist()
        N_x = Counter(map(tuple, raw_queries))

        f_xa: dict[tuple[int, ...], float] = {}
        for xa in N_xa:
            x = xa[:self.players]
            f_xa[xa] = N_xa.get(xa, 0) / N_x.get(x, 0)
        
        return f_xa

def pvalue_score(y: np.ndarray):
    # Any entries where y = 0 imply that f(a|x) = 0, which means it wasn't
    # in our training data. Todo: is this the best strategy to handle it?
    log_y = np.log(y[y > 0])
    log_t = log_y.sum()
    t = np.exp(log_t)
    return min(t ** -1, 1)
