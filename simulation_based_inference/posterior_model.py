import torch
from sbi.inference import NPE
from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from encoding_model import ConvNet, MLP
## normalizing flow, diffusion, or other probabilistic models as the density estimator

## using 2 different bind/unbind rates for site 2, site 4
## we'd probably want to use a single rate for a single site, but maybe we can get that from fitting
low = torch.tensor([-3, 0, -4, -1, -4, -1, 10])
high = torch.tensor([1, 3, -1, 2, -1, 2, 100])

theta = torch.tensor([])
x = torch.tensor([])

prior = utils.torchutils.BoxUniform(low=low, high=high)

encoding_net = ConvNet(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    output_dim=32,
    padding=1,
)

# using MAF density estimator
density_estimator = posterior_nn(model="zuko_maf", embedding_net=encoding_net)

inference = NPE(prior=prior, density_estimator=density_estimator)

posterior = inference.append_simulations(theta, x).train()

posterior_dist = inference.build_posterior()

