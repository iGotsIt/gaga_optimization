import torch
from sbi.inference import NPE
from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from encoding_model import ConvNet, MLP
from torch.distributions import Normal
## normalizing flow, diffusion, or other probabilistic models as the density estimator

## using 2 different bind/unbind rates for site 2, site 4
## we'd probably want to use a single rate for a single site, but maybe we can get that from fitting

device = "cuda" if torch.cuda.is_available() else "cpu"
low = torch.tensor([-3, 0, -4, -1, -4, -1, 10], device=device)
high = torch.tensor([1, 3, -1, 2, -1, 2, 100], device=device)

theta = torch.tensor([])
x = torch.tensor([])


prior = utils.torchutils.BoxUniform(low=low, high=high)

prior = utils.MultipleIndependent(
    [
        utils.torchutils.BoxUniform(low[0:2], high[0:2]),  # first 2 parameters
        Normal(torch.tensor([-2.0056]), torch.tensor([0.6055])),  # middle 2 params
        Normal(torch.tensor([0.7622]), torch.tensor([0.5725])),
        utils.torchutils.BoxUniform(low[4:7], high[4:7]),  # final 3 parameters
    ]
)  ## build prior from what the data distribution looks like


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

inference = NPE(prior=prior, density_estimator=density_estimator, device=device)

posterior = inference.append_simulations(theta, x, data_device="cpu").train()

posterior_dist = inference.build_posterior()