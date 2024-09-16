import torch 
import numpy as np 
import torch.nn.functional as F 

def compute_rays(img, pose, focal, low_dist, far_dist, nc, device):
    batch_size,h,w,_ = img.shape
    x = torch.linspace(0, w - 1, w, device=device)
    y = torch.linspace(0, h - 1, h, device=device)
    xi, yi = torch.meshgrid(x, y, indexing="xy")

    xi = xi.unsqueeze(0).repeat(batch_size,1,1)
    yi = yi.unsqueeze(0).repeat(batch_size,1,1)    

    focal = focal.view(-1,1,1)

    norm_x = (xi - w * 0.5) / focal.to(device)
    norm_y = (yi - h * 0.5) / focal.to(device)

    direction = torch.stack([norm_x, -norm_y, -torch.ones_like(norm_x)], dim=-1)
    rotation = pose[:,:3,:3].float()
    translation = pose[:,:3,-1]
    translation = translation.unsqueeze(1)
    direction_reshaped = direction.view(batch_size, h*w, 3).float()
    # print(direction_reshaped.shape)
    # print(direction_reshaped, rotation)
    ray_direction = torch.bmm(direction_reshaped, rotation.transpose(1, 2))
    ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)

    ray_origins = translation.expand_as(ray_direction)

    depth_vals = torch.linspace(low_dist, far_dist, nc, device=device)
    noise = torch.rand((batch_size, h*w,nc), device=device) * (far_dist - low_dist) / nc

    depth_vals = depth_vals + noise
    # print(f"depth values {depth_vals.shape}")
    
    
    query_points = ray_origins[..., None, :] + ray_direction[..., None, :] * depth_vals[..., :, None]
    # print(f"query points {query_points.shape}\ndetpth:{depth_vals.shape}")

    depth_vals = depth_vals.reshape(batch_size, 100, 100, depth_vals.shape[2])
    query_points = query_points.reshape(batch_size, 100, 100, query_points.shape[2], 3)
    # print(f"query points {query_points.shape}")
    # print()
    return ray_direction, ray_origins, depth_vals, query_points


def position_encoding(points, num_encode):
    gamma = [points]
    for i in range(num_encode):
        gamma.append(torch.sin((2.0**i) * points))
        gamma.append(torch.cos((2.0**i) * points))
    
    gamma = torch.cat(gamma, axis=-1)
    return gamma


def render(predictions, query_points, ray_origins, depth_values):
    radiance_field_flat = predictions
    unflat_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flat, unflat_shape)

    sigma_a = F.relu(radiance_field[...,3])       #volume density
    rgb = torch.sigmoid(radiance_field[...,:3])    #color value at nth depth value
    one_e_10 = torch.tensor([1e10], dtype = ray_origins.dtype, device = ray_origins.device)
    dists = torch.cat((depth_values[...,1:] - depth_values[...,:-1], one_e_10.expand(depth_values[...,:1].shape)), dim = -1)
    
    alpha = 1. - torch.exp(-sigma_a * dists)       
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)     #transmittance
    rgb_map = (weights[..., None] * rgb).sum(dim = -2)          #resultant rgb color of n depth values
    depth_map = (weights * depth_values).sum(dim = -1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map


def cumprod_exclusive(tensor) :
  dim = -1
  cumprod = torch.cumprod(tensor, dim)
  cumprod = torch.roll(cumprod, 1, dim)
  cumprod[..., 0] = 1.
  
  return cumprod