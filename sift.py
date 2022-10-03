
from __future__ import annotations
from typing import Tuple
import torch
import cv2
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_keypoints
from PIL import Image
import math

def detect(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
    image = image.transpose(0, -1)
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)

    descriptors = torch.from_numpy(descriptors)
    kps = []
    for kp in keypoints:
        pt = kp.pt
        kps.append((pt[1], pt[0]))
        
    keypoints = torch.tensor(kps)
    
    return keypoints, descriptors

def _chunk_topk(X, k, largest=True, chunk_size=64):
    X = X.split(chunk_size)
    dists, indices = [], []
    for i in range(len(X)):
        x = X[i]
        x.to('cuda', non_blocking=True)
        dist, index = torch.topk(x, k, dim=-1, largest=largest)
        dists.append(dist.to('cpu'))
        indices.append(index.to('cpu'))
        
    return torch.cat(dists, dim=0), torch.cat(indices, dim=0)



def pairwise_match(index, query, thr=0.7):
    '''
    index: (N, D1, F) N: number of images, D: number of descriptors, F: descriptor dimension
    query: (M, D2, F) M: number of queries, D: number of descriptors, F: descriptor dimension
    thr: ratio test threshold
    
    idx_idesc_qdesc: (K, 4) K: number of matches, 4: (index image, index descriptor, query image, query descriptor)
    '''
    sum_ = torch.sum(index)
    sum_ += torch.sum(query)
    mean_ = sum_ / (torch.numel(index) + torch.numel(query))

    # (N, M, D1, D2)  d_matrix[i, j, k, l] = distance between index[i, k] and query[j, l]
    d_matrix = torch.cdist(index.unsqueeze(1), query.unsqueeze(0)) 

    # (N, M, D1, 2)  indices[i, j, k, l] = m -> index image i with descriptor k has top the lth smallest distances with query image j with descriptor m
    dists, indices = _chunk_topk(d_matrix, 2, largest=False) 
    
    # (N, M, D1)  matches[i, j, k] = True if index image i with descriptor j passed the ratio test with query image k
    matches = dists[..., 0] < thr * dists[..., 1] # (N, D)

    print(matches.shape)
    
    # (K, 3) idx_idesc_qdesc[i] = (j, k, l) -> the ith matched descriptor is from index image j with descriptor k and query image l
    i_q_idesc = torch.argwhere(matches)
    print(matches.shape)
    # (K, 1) qdesc[i] = m  -> the ith matched descriptor is from query image l with descriptor m
    qdesc = indices[matches][:, [0]]

    # (M, 4) idx_idesc_qdesc[i] = (j, k, l, m) -> the ith matched descriptor is from index image j with descriptor k and query image l with descriptor m
    i_q_idesc_qdesc = torch.cat([i_q_idesc, qdesc], dim=1) 
    return i_q_idesc_qdesc



def listwise_match(index, query, thr=0.7):
    '''
    index: (N, D1, F) N: number of images, D: number of descriptors, F: descriptor dimension
    query: (N, D2, F) N: number of queries, D: number of descriptors, F: descriptor dimension
    thr: ratio test threshold
    
    idx_idesc_qdesc: (M, 3) M: number of matches, 3: (listwise index, index descriptor, query descriptor)
    '''
    sum_ = torch.sum(index)
    sum_ += torch.sum(query)
    mean_ = sum_ / (torch.numel(index) + torch.numel(query))

    d_matrix = torch.cdist(index, query) # (i, j, k)  d_matrix[i, j, k] = distance between index[i, j] and query[i, k]

    # (N, D, 2)  indices[i, j, k] = m -> index image i with descriptor j has top the kth smallest distances with query image i with descriptor m
    dists, indices = _chunk_topk(d_matrix, 2, largest=False) 
    # (N, D)  matches[i, j] = True if index image i with descriptor j passed the ratio test
    matches = dists[..., 0] < thr * dists[..., 1] # (N, D)
    n_matches = torch.sum(matches, dim=-1) # (N, ) number of matches for each image
    full_rank = n_matches >= 4 # (N, ) True if image has at least 4 matches
    possible = matches & full_rank.unsqueeze(-1) # (N, D) True if image has at least 4 matches and descriptor j passed the ratio test
    
    # (M, 2) idx_idesc[i] = (j, k) -> the ith matched descriptor is from index image j with descriptor k
    idx_idesc = torch.argwhere(possible)

    # (M, 1) idx_qdesc[i] = k  -> the ith matched descriptor is from query image j with descriptor k (j is the same as idx_idesc[i, 0])
    qdesc = indices[possible][:, [0]]

    # (M, 3) idx_idesc_qdesc[i] = (j, k, l) -> the ith matched descriptor is from index image j with descriptor k and query image j with descriptor l
    idx_idesc_qdesc = torch.cat([idx_idesc, qdesc], dim=1) 
    return idx_idesc_qdesc


@torch.jit.script
def ikp_qkp(ikps, qkps, idx_idesc_qdesc):


    unique_elms = torch.unique(idx_idesc_qdesc[:, 0])
    max_points = max([len(idx_idesc_qdesc[idx_idesc_qdesc[:, 0] == i]) for i in unique_elms])

    # (N, M, 2)  kp_pairs[i, j] = (k, l) -> the i the mathched image has M many matches, of witch the jth matched keypoints has coordinates (k, l)
    idx_ikp = torch.zeros((len(unique_elms), max_points, 2))
    idx_qkp = torch.zeros((len(unique_elms), max_points, 2))
    for i, pair_idx in enumerate(unique_elms):
        idesc_qdesc = idx_idesc_qdesc[idx_idesc_qdesc[:, 0] == pair_idx]
        ikp = ikps[pair_idx][idesc_qdesc[:, 1]]
        qkp = qkps[pair_idx][idesc_qdesc[:, 2]]
        idx_ikp[i, :len(ikp)] = ikp
        idx_qkp[i, :len(qkp)] = qkp
    return idx_ikp, idx_qkp


@torch.jit.script
def ransac(kps1, kps2, its: int = 32, ratio: float = 0.6):
    
    B, N, F = kps1.shape
    
    rasac_size = math.ceil(N * ratio)

    perm = [torch.randperm(N) for _ in range(its)]
    perm = torch.stack(perm)

    perm = perm[:, :rasac_size].cuda()
    

    pad = torch.ones((B, N, 1))

    X = torch.cat([kps1, pad], dim=-1).cuda()
    Y = torch.cat([kps2, pad], dim=-1).cuda()

    sample_X = X[:, perm] 
    sample_Y = Y[:, perm] # ransac_its x rasac_size x 3

    
    H, _, _, _ = torch.linalg.lstsq(sample_X, sample_Y) # frobinous norm, it is equivalent to minimize ||AX - Y||_F for each X

    YH = torch.matmul(X, H)

    error = torch.norm(Y - YH, 2, dim=(-1, -2))

    min_index = torch.argmin(error, dim=-1)
    
    H = H[:, min_index]
    H = H.view(-1, 3, 3)

    
    error = torch.norm(Y - X @ H, 2, dim=-1)


    thr = torch.mean(error, dim=-1, keepdim=True)
    inliers = error < thr

    selected_X = X[inliers]
    selected_Y = Y[inliers]

    selected_X = selected_X[:, :2] / selected_X[:, 2:]
    selected_Y = selected_Y[:, :2] / selected_Y[:, 2:]

    return H, selected_X, selected_Y


def visualize_keypoints(image, keypoints, color='red', radius=2):
    image = draw_keypoints(image, keypoints, colors=color, radius=radius)
    image = to_pil_image(image)
    return image

def draw_match_lines(image1, image2, kps1, kps2):
    image1 = image1.cpu()
    image2 = image2.cpu()
    kps1 = kps1.cpu()
    kps2 = kps2.cpu()
    kps2 = kps2 + torch.tensor([image1.shape[-1], 0])
    all_kps = torch.cat([kps1, kps2], dim=0)
    all_kps = all_kps.unsqueeze(0)
    connectivity = torch.stack([torch.arange(len(kps1)), torch.arange(len(kps2)) + len(kps1)], dim=1)
    connectivity = connectivity.tolist()
    
    side_by_side = torch.cat([image1, image2], dim=-1)
    image = draw_keypoints(side_by_side, all_kps, connectivity, colors='red', radius=2)
    image = to_pil_image(image)
    return image

def draw_transfrom_points(index_image, query_image, H, selected_X, selected_Y):
    
    padded_X = torch.cat([selected_X, torch.ones(selected_X.shape[0], 1, device=selected_X.device)], dim=1)
    
    projected_X = torch.matmul(padded_X, H)
    projected_X = projected_X[:,...,:2] / projected_X[:,...,2:]
    

    index_image = visualize_keypoints(index_image, selected_X.unsqueeze(0))
    query_image = visualize_keypoints(query_image, selected_Y.unsqueeze(0))
    query_image = pil_to_tensor(query_image)

    query_image = visualize_keypoints(query_image, projected_X, color='blue')
    side_by_side = concat(index_image, query_image, dim=1)
    return side_by_side

def concat(im1, im2, dim=1):
    dest = None
    if dim == 1:
        dest = Image.new('RGB', (im1.width + im2.width, im1.height))
        dest.paste(im1, (0, 0))
        dest.paste(im2, (im1.width, 0))
    else:
        dest = Image.new('RGB', (im1.width, im1.height + im2.height))
        dest.paste(im1, (0, 0))
        dest.paste(im2, (0, im1.height))
    return dest
