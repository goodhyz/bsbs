import os

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .networks import MultiHeadEncoder
from .rldeafl import Feature_Extractor, Critic, RLDEAFL, mySequential
from ...rl.ppo import PPO_Agent


# ------------------------------------------------------------------ #
#  Transformer Actor                                                  #
#                                                                     #
#  Keeps RLDE-AFL's action space but replaces the six independent     #
#  two-layer MLP heads with a *shared* Transformer encoder followed   #
#  by single linear output heads.                                     #
# ------------------------------------------------------------------ #
class TransformerActor(nn.Module):
    """
    # Introduction
    Transformer-based policy head used by ``RLDEAFL_T``.

    The original RLDE-AFL actor is composed of six independent two-layer
    MLP heads, each operating *per individual* and therefore unable to
    directly model inter-individual relationships.  Inspired by GLEET,
    which uses a Transformer self-attention module to capture such
    relationships, this actor wraps a shared Transformer encoder
    (multi-head self-attention over the population dimension) around
    single linear output heads.

    # Action Space (identical to RLDE-AFL)
    - mutation operator selection   : Categorical over ``mu_operator``
    - crossover operator selection  : Categorical over ``cr_operator``
    - continuous mutation params F  : Normal of dimension ``n_mutation``
    - continuous crossover params Cr: Normal of dimension ``n_crossover``

    # Input / Output
    - input  : ``[bs, NP, input_dim]`` per-individual features produced
               by the NeurELA feature extractor.
    - action : ``[bs, NP, 2 + n_mutation + n_crossover]`` (same layout
               as RLDE-AFL's ``Actor``).
    """

    def __init__(self,
                 input_dim,
                 mu_operator,
                 cr_operator,
                 n_mutation,
                 n_crossover,
                 n_heads=4,
                 n_encoder_layers=2,
                 feed_forward_hidden=None):
        super().__init__()

        if feed_forward_hidden is None:
            feed_forward_hidden = input_dim * 2

        self.encoder = mySequential(*(
            MultiHeadEncoder(n_heads=n_heads,
                             embed_dim=input_dim,
                             feed_forward_hidden=feed_forward_hidden,
                             normalization='layer')
            for _ in range(n_encoder_layers)
        ))

        self.mutation_selector_head = nn.Linear(input_dim, mu_operator)
        self.crossover_selector_head = nn.Linear(input_dim, cr_operator)
        self.mutation_param_mu_head = nn.Linear(input_dim, n_mutation)
        self.mutation_param_sigma_head = nn.Linear(input_dim, n_mutation)
        self.crossover_param_mu_head = nn.Linear(input_dim, n_crossover)
        self.crossover_param_sigma_head = nn.Linear(input_dim, n_crossover)

        self.max_sigma = 0.7
        self.min_sigma = 0.1

        self.n_mutation = n_mutation
        self.n_crossover = n_crossover

    def get_parameter_number(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total, 'Trainable': trainable}

    def _compute_distributions(self, x):
        h = self.encoder(x)

        m_logits = self.mutation_selector_head(h)
        c_logits = self.crossover_selector_head(h)
        m_probs = torch.softmax(m_logits, dim=2)
        c_probs = torch.softmax(c_logits, dim=2)
        m_selector = Categorical(m_probs)
        c_selector = Categorical(c_probs)

        m_mu = (torch.tanh(self.mutation_param_mu_head(h)) + 1.) / 2.
        m_sigma = (torch.tanh(self.mutation_param_sigma_head(h)) + 1.) / 2. \
                  * (self.max_sigma - self.min_sigma) + self.min_sigma
        c_mu = (torch.tanh(self.crossover_param_mu_head(h)) + 1.) / 2.
        c_sigma = (torch.tanh(self.crossover_param_sigma_head(h)) + 1.) / 2. \
                  * (self.max_sigma - self.min_sigma) + self.min_sigma

        m_policy = Normal(m_mu, m_sigma)
        c_policy = Normal(c_mu, c_sigma)

        return m_selector, c_selector, m_policy, c_policy, (m_mu, m_sigma, c_mu, c_sigma)

    def get_action(self, x):
        m_sel, c_sel, m_policy, c_policy, (m_mu, m_sigma, c_mu, c_sigma) = \
            self._compute_distributions(x)

        m_op = m_sel.sample()
        c_op = c_sel.sample()
        m_act = torch.clamp(m_policy.sample(), min=0, max=1)
        c_act = torch.clamp(c_policy.sample(), min=0, max=1)

        action = torch.cat([
            m_op[:, :, None],
            c_op[:, :, None],
            m_act,
            c_act,
        ], dim=2)

        return (action, m_mu, m_sigma, c_mu, c_sigma)

    def forward(self, x, fixed_action=None, require_entropy=False):
        m_sel, c_sel, m_policy, c_policy, _ = self._compute_distributions(x)

        if fixed_action is not None:
            m_op = fixed_action[:, :, 0]
            c_op = fixed_action[:, :, 1]
            m_act = fixed_action[:, :, 2:2 + self.n_mutation]
            c_act = fixed_action[:, :, -self.n_crossover:]
        else:
            m_op = m_sel.sample()
            c_op = c_sel.sample()
            m_act = torch.clamp(m_policy.sample(), min=0, max=1)
            c_act = torch.clamp(c_policy.sample(), min=0, max=1)

        action = torch.cat([
            m_op[:, :, None],
            c_op[:, :, None],
            m_act,
            c_act,
        ], dim=2)

        m_op_lp = m_sel.log_prob(m_op)
        c_op_lp = c_sel.log_prob(c_op)
        m_lp = m_policy.log_prob(m_act).sum(dim=2)
        c_lp = c_policy.log_prob(c_act).sum(dim=2)
        log_prob = (m_op_lp + c_op_lp + m_lp + c_lp).sum(dim=1)

        if require_entropy:
            entropy = torch.cat([
                m_sel.entropy()[:, :, None],
                c_sel.entropy()[:, :, None],
                m_policy.entropy(),
                c_policy.entropy(),
            ], dim=2)
            return (action, log_prob, entropy)

        return (action, log_prob)


# ------------------------------------------------------------------ #
#  RLDEAFL_T  (RLDE-AFL with a Transformer policy head)               #
# ------------------------------------------------------------------ #
class RLDEAFL_T(RLDEAFL):
    """
    # Introduction
    ``RLDEAFL_T`` is the improved algorithm proposed in Chapter 4 of the
    thesis.  It keeps RLDE-AFL's NeurELA-based feature extractor and its
    training procedure unchanged, and replaces the original six
    independent two-layer MLP heads in the actor with a shared
    Transformer encoder followed by single linear output heads.

    # Motivation
    In RLDE-AFL, NeurELA extracts a per-individual feature vector, but
    each of the six MLP heads then decides the mutation / crossover
    operator and the continuous F / Cr parameters by looking only at
    that individual's own feature vector.  Inter-individual
    relationships are therefore captured only implicitly by the
    feature extractor.  GLEET shows that a Transformer self-attention
    module over the population is an effective way to capture such
    relationships.  ``RLDEAFL_T`` combines the two: NeurELA provides
    automated feature learning, and a Transformer policy head
    explicitly models inter-individual dependencies before making
    per-individual decisions.

    # Architecture Summary
    - Feature extractor : NeurELA (``Feature_Extractor`` from RLDE-AFL)
    - Actor             : Transformer encoder (stack of
      ``MultiHeadEncoder`` layers) + six linear output heads
      (``TransformerActor``)
    - Critic            : MLP with mean-pooling over individuals (the
      same ``Critic`` as RLDE-AFL)

    # Action Space (identical to RLDE-AFL)
    - 14 mutation operators
    - 3  crossover operators
    - 3-dim continuous mutation parameters (including F)
    - 2-dim continuous crossover parameters (including Cr)

    # Training
    Proximal Policy Optimization with a clipped surrogate objective.
    The training loop is inherited verbatim from
    ``baseline.metabbo.rldeafl.RLDEAFL``.

    # Application Scenario
    Single-objective continuous black-box optimisation (SSO).

    # Args
    - config: Configuration namespace (see ``config.py``).
    """

    def __init__(self, config):
        self.config = config

        self.config.optimizer = 'Adam'
        self.config.lr = 1e-4

        self.config.fe_hidden_dim = 64
        self.config.fe_n_layers = 1

        self.config.gamma = 0.99
        self.config.n_step = 10
        self.config.K_epochs = 3
        self.config.eps_clip = 0.2
        self.config.max_grad_norm = 1

        self.config.mu_operator = 14
        self.config.cr_operator = 3
        self.config.n_mutation = 3
        self.config.n_crossover = 2

        self.config.actor_n_heads = 4
        self.config.actor_n_encoder_layers = 2

        actor_input_dim = self.config.fe_hidden_dim + 16

        fe = Feature_Extractor(
            hidden_dim=self.config.fe_hidden_dim,
            n_layers=self.config.fe_n_layers,
            device=self.config.device,
        )
        actor = TransformerActor(
            input_dim=actor_input_dim,
            mu_operator=self.config.mu_operator,
            cr_operator=self.config.cr_operator,
            n_mutation=self.config.n_mutation,
            n_crossover=self.config.n_crossover,
            n_heads=self.config.actor_n_heads,
            n_encoder_layers=self.config.actor_n_encoder_layers,
        )
        critic = Critic(input_dim=actor_input_dim)

        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name,
        )

        # NOTE: skip ``RLDEAFL.__init__`` (which would build vanilla
        # RLDE-AFL's MLP actor) and hand our custom modules directly
        # to ``PPO_Agent``; ``train_episode`` and ``rollout_episode``
        # are inherited from ``RLDEAFL`` unchanged.
        PPO_Agent.__init__(
            self,
            self.config,
            {'actor': actor, 'critic': critic, 'fe': fe},
            self.config.lr,
        )

    def __str__(self):
        return "RLDEAFL_T"
