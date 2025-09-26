from .q_learner import QLearner, AuxQLearner, ContQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .DRMAC_learner import DRMACLearner
from .DRMAC_qplex_learner import DRMACLearner as DRMACQPLEXLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["aux_q_learner"] = AuxQLearner  # customized
REGISTRY["cont_q_learner"] = ContQLearner  # customized
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["DRMAC_learner"] = DRMACLearner
REGISTRY["DRMAC_qplex_learner"] = DRMACQPLEXLearner