from .rldeafl_optimizer import RLDEAFL_Optimizer


class RLDEAFL_T_Optimizer(RLDEAFL_Optimizer):
    """
    # Introduction
    Low-level DE optimiser driven by the ``RLDEAFL_T`` meta-layer
    policy.  Its action space, operator pool, population update
    procedure and reward definition are identical to
    ``RLDEAFL_Optimizer`` because the improvement proposed in Chapter 4
    of the thesis only touches the policy network architecture: the
    NeurELA-based feature extractor's six independent MLP heads are
    replaced by a shared Transformer encoder followed by linear heads.
    Since the DE operators and their hyper-parameter semantics are not
    modified, the optimiser simply inherits from
    ``RLDEAFL_Optimizer`` and only changes the string identifier used
    for logging and checkpoint directories.

    # Action Layout (per individual, inherited from RLDE-AFL)
    - ``action[:, 0]``            : mutation operator index (0 .. 13)
    - ``action[:, 1]``            : crossover operator index (0 .. 2)
    - ``action[:, 2 : 2 + 3]``    : continuous mutation parameters (F, ...)
    - ``action[:, -2:]``          : continuous crossover parameters (Cr, ...)

    # Mutation Pool (14 operators, see ``rldeafl_optimizer.py``)
    best/1, best/2, rand/1, rand/2, current-to-best/1, rand-to-best/1,
    current-to-rand/1, current-to-pbest/1, ProDE-rand/1, TopoDE-rand/1,
    current-to-pbest/1+archive, HARDDE-current-to-pbest/2,
    current-to-rand/1+archive, weighted-rand-to-pbest/1.

    # Crossover Pool (3 operators)
    binomial, exponential, p-binomial.
    """

    def __str__(self):
        return "RLDEAFL_T_Optimizer"
