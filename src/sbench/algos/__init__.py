from .adversarial import AdversarialPG
from .base import Algo, BudgetExhausted, Evaluator
from .bayesopt import BayesOpt
from .ce_is import CrossEntropyIS
from .nsga2 import NSGA2
from .sampling import LHS, GridSearch, RandomSearch

ALGOS = {
    "random": RandomSearch,
    "grid": GridSearch,
    "lhs": LHS,
    "bayesopt": BayesOpt,
    "nsga2": NSGA2,
    "adv_pg": AdversarialPG,
    "ce_is": lambda: CrossEntropyIS(screen_k=1),
    "surrogate_ce": lambda: CrossEntropyIS(screen_k=10),
}


def make_algo(name: str) -> Algo:
    return ALGOS[name]()
