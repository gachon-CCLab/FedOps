from omegaconf import OmegaConf
from fedops.client.client_utils import gen_parameter_shape

cfg = OmegaConf.load("./conf/config.yaml")
gen_parameter_shape(cfg)