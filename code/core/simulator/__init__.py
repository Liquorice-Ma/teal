from . import util

from .se import SplitEqual
from .ls import LocalSearch
from .sp import ShortestPath
from .eslp import ElephantSplitLinearProgram
from .ceslp import ClusteredElephantSplitLinearProgram

def create_solver(args):
    if args.solver == 'sp':
        solver = ShortestPath(args)
        return solver
    elif args.solver == 'se':
        solver = SplitEqual(args)
        return solver
    elif args.solver == 'eslp':
        solver = ElephantSplitLinearProgram(args)
        return solver
    elif args.solver == 'ceslp':
        solver = ClusteredElephantSplitLinearProgram(args)
        return solver
    elif args.solver == 'ls':
        solver = LocalSearch(args)
        return solver
    else:
        raise NotImplementedError
