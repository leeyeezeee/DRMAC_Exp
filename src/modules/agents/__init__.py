REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .ExpoComm_agent import (
    ExpoCommSAgent,
    ExpoCommOAgent,
    ExpoCommSContAgent,
    ExpoCommOContAgent,
)

from .DRMAC_agent import DRMACAgent

REGISTRY["ExpoComm_static"] = ExpoCommSAgent
REGISTRY["ExpoComm_one_peer"] = ExpoCommOAgent
REGISTRY["ExpoComm_static_cont"] = ExpoCommSContAgent
REGISTRY["ExpoComm_one_peer_cont"] = ExpoCommOContAgent
REGISTRY["DRMAC"] = DRMACAgent