from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO, List, Tuple

import cv2
import numpy as np
import torch
from torch import jit

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


def get_keypoints_descriptors(data: str | torch.Tensor):
    keypoints, descriptors = detect(data)
    return keypoints, descriptors


def detect(
    image_like: torch.Tensor | np.ndarray | str | Path | bytes | BinaryIO,
    input_shape: str = "CHW",
    input_channels: str = "RGB",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    image_like: (C, H, W) or (H, W, C) or path to image or bytes or file-like object
    input_shape: 'CHW' or 'HWC', the shape of the input image
    input_channels: 'RGB' or 'BGR', the order of the channels in the input image

    return:
    keypoints: (N, 2) N: number of keypoints, 2: (x, y) coordinates
    descriptors: (N, 128) N: number of keypoints, 128: descriptor
    """
    input_shape = input_shape.upper()
    input_channels = input_channels.upper()
    # shapes: Any to (HWC)
    if isinstance(image_like, torch.Tensor):
        image_like = image_like.numpy()

    if isinstance(image_like, (str, Path)):
        image_like = cv2.imread(str(image_like), cv2.IMREAD_COLOR)
        input_shape = "HWC"
        input_channels = "BGR"
    if isinstance(image_like, bytes):
        buff = io.BytesIO(image_like)
        image_like = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        input_shape = "HWC"
        input_channels = "BGR"
    if isinstance(image_like, BinaryIO):
        image_like = cv2.imdecode(image_like, cv2.IMREAD_COLOR)
        input_shape = "HWC"
        input_channels = "BGR"

    assert isinstance(image_like, np.ndarray), "image_like must be a numpy array"
    if input_shape == "CHW":
        image_like = np.swapaxes(image_like, 0, 2)  # (C, H, W) -> (W, H, C)
        image_like = np.swapaxes(image_like, 0, 1)  # (W, H, C) -> (H, W, C)

    if input_shape == "CWH":
        image_like = np.swapaxes(image_like, 0, 2)  # (C, W, H) -> (W, H, C)
        image_like = np.swapaxes(image_like, 0, 1)  # (W, H, C) -> (H, W, C)

    if input_shape == "WHC":
        image_like = image_like.swapaxes(0, 1)  # (W, H, C) -> (H, W, C)

    image_like = image_like.astype(np.uint8)
    if input_channels == "RGB":
        image_like = cv2.cvtColor(image_like, cv2.COLOR_RGB2BGR)

    image_like = cv2.cvtColor(image_like, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image_like, None)

    descriptors = torch.from_numpy(descriptors)
    kps = []
    for kp in keypoints:
        pt = kp.pt
        kps.append((pt[1], pt[0]))

    keypoints = torch.tensor(kps)
    if input_shape == "CHW":
        keypoints = keypoints.flip(-1)
    return keypoints, descriptors


@jit.script
def sample(
    kpts_list: List[torch.Tensor], dtrs_list: List[torch.Tensor], size: int = 384
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    kpts_list: a list of keypoints, where kpts_list[i] is a tensor of shape (N_i, 2)
    dtrs_list: a list of descriptors, where dtrs_list[i] is a tensor of shape (N_i, F)

    returns:
    KP: (B, size, 2) tensor of keypoints
    DT: (B, size, F) tensor of descriptors
    """
    # uniform sampling to get desired number of descriptors
    N = len(kpts_list)

    PD = kpts_list[0].size(-1)
    DF = dtrs_list[0].size(-1)

    KP = torch.empty((N, size, PD), device=kpts_list[0].device)
    DT = torch.empty((N, size, DF), device=dtrs_list[0].device)

    for idx in range(N):
        kpts = kpts_list[idx]
        dtrs = dtrs_list[idx]
        n_kpts = len(kpts)

        index = (
            torch.randperm(size) % n_kpts
            if size > n_kpts
            else torch.randperm(n_kpts)[:size]
        )

        KP[idx] = kpts[index]
        DT[idx] = dtrs[index]

    return KP, DT


@jit.script
def _min_max_norm(
    index: torch.Tensor, query: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_val = torch.min(torch.min(index), torch.min(query))
    max_val = torch.max(torch.max(index), torch.max(query))

    index = (index - min_val) / (max_val - min_val + 1e-12)
    query = (query - min_val) / (max_val - min_val + 1e-12)
    return index, query


@jit.script
def match(
    index: torch.Tensor, query: torch.Tensor, thr: float = 0.7, full_rank_thr: int = 8
) -> torch.Tensor:
    """
    index: (N, K, F) N: number of images, K: number of descriptors, F: descriptor dimension
    query: (N, K, F) N: number of queries, K: number of descriptors, F: descriptor dimension
    thr: ratio test threshold

    idx_idesc_qdesc: (M, 3) M: number of matches, 3: (listwise index, index descriptor, query descriptor)
    """
    # normalize
    index = index.double()
    query = query.double()
    index, query = _min_max_norm(index, query)
    # compute distance matrix of each batch
    # since I is in shape (N, K, F)
    # and Q is in shape  (N, K, F),
    # the distance matrix is in shape (N, K, K)
    # where distance is caluated batchwise

    D = torch.cdist(index.double(), query.double())

    # find the top 2 smallest distances for each descriptor
    TD, TI = torch.topk(D, 2, dim=-1, largest=False)
    # TD and TI are both in shape (N, K, 2)

    # ratio test to find matches
    # M is in shape (N, K)
    M = TD[..., 0] <= thr * TD[..., 1]

    # compute the number of matches for each batch
    # NM is in shape (N, )
    NM = torch.sum(M, dim=-1)
    # for each batch, check if the number of matches
    # is greater than the threshold of minimum number of matches
    full_rank = NM >= full_rank_thr
    # if a batch has less than the threshold of minimum number of matches
    # then maskout all the matches for that batch to be False
    # valid is in shape (N, K)
    valid = M & full_rank.unsqueeze(-1)

    # find the indices of index image and its descriptors for each matched pair
    IDX1 = torch.argwhere(valid)
    # find the indices of query image and its descriptors for each matched pair
    IDX2 = TI[valid]
    # get the closest descriptor for each matched pair
    IDX2 = IDX2[..., 0].unsqueeze(-1)
    # concatenate
    IDX = torch.cat([IDX1, IDX2], dim=-1)

    return IDX


@jit.script
def inflate(
    IKP: torch.Tensor, QKP: torch.Tensor, IDX: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    IKP: (N, D1, 3) N: number of images, D: number of descriptors, 3: (x, y, scale)
    QKP: (N, D2, 3) N: number of queries, D: number of descriptors, 3: (x, y, scale)
    IDX: (M, 3) M: number of matches, 3: (listwise index, index descriptor, query descriptor)
    max_: maximum number of matches per image

    return:
    IIKP: (N, max_, 3) N: number of images, max_: max number of matches, 3: (x, y, scale)
    IQKP: (N, max_, 3) N: number of queries, max_: max number of matches, 3: (x, y, scale)
    """
    # unique listwise indices
    UE, counts = torch.unique(IDX[:, 0], return_counts=True)
    # max number of matches per image
    max_ = torch.amax(counts)
    # build empty containers for inflated index keypoints and query keypoints
    IIKP = torch.zeros((len(UE), int(max_), 2))  # inflated index keypoints
    IQKP = torch.zeros((len(UE), int(max_), 2))  # inflated query keypoints

    for i, idx in enumerate(UE):
        # get all matches for image idx
        pair = IDX[IDX[:, 0] == idx]
        # get index keypoints and query keypoints
        ikp = IKP[idx][pair[:, 1]]
        qkp = QKP[idx][pair[:, 2]]
        # fill into containers
        IIKP[i, : len(ikp)] = ikp
        IQKP[i, : len(qkp)] = qkp
    return IIKP, IQKP
