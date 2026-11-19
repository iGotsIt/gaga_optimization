from posterior_model import posterior, posterior_dist
import torch


x_observation = torch.tensor([])
posterior_params = posterior.sample(torch.Size([1]), x_observation)

