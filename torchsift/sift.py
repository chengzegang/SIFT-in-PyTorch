from __future__ import annotations
from pathlib import Path
from typing import BinaryIO, List, Literal, Tuple
import torch
import cv2
import numpy as np
import io
import logging

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


def get_keypoints_descriptors(data: str | torch.Tensor):
    keypoints, descriptors = detect(data)
    return keypoints, descriptors


def detect(
    image_like: torch.Tensor | np.ndarray | str | Path | bytes | BinaryIO,
    input_shape: Literal["CHW", "HWC"] = "CHW",
    input_channels: Literal["RGB", "BGR"] = "RGB",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    image_like: (C, H, W) or (H, W, C) or path to image or bytes or file-like object
    input_shape: 'CHW' or 'HWC', the shape of the input image
    input_channels: 'RGB' or 'BGR', the order of the channels in the input image

    return:
    keypoints: (N, 2) N: number of keypoints, 2: (x, y) coordinates
    descriptors: (N, 128) N: number of keypoints, 128: descriptor
    """

    image = None
    if isinstance(image_like, torch.Tensor):
        if input_shape == "CHW":
            image_like = image_like.transpose(0, 2)
        image = image_like.numpy()
    if isinstance(image_like, np.ndarray):
        if input_shape == "CHW":
            image_like = image_like.transpose(0, 2)
        image = image_like
    if isinstance(image_like, (str, Path)):
        image = cv2.imread(image_like, cv2.IMREAD_COLOR)
        input_channels = "BGR"
    if isinstance(image_like, bytes):
        buff = io.BytesIO(image_like)
        image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    if isinstance(image_like, BinaryIO):
        image = cv2.imdecode(image_like, cv2.IMREAD_COLOR)

    image = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if input_channels == "RGB"
        else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )

    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)

    descriptors = torch.from_numpy(descriptors)
    kps = []
    for kp in keypoints:
        pt = kp.pt
        kps.append((pt[1], pt[0]))

    keypoints = torch.tensor(kps)

    return keypoints, descriptors


@torch.jit.script
def topk_with_chunk(
    X: torch.Tensor,
    k: torch.Tensor,
    largest: torch.Tensor = torch.Tensor([1]),
    chunk_size: torch.Tensor = torch.tensor([512]),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: (N, D) N: number of elements, D: dimension
    k: (1) number of elements to select
    largest: (1) 1: largest, 0: smallest
    chunk_size: (1) number of elements to process at a time

    returns:
    D: (k, D) k: number of elements to select, D: dimension
    I: (k) k: number of elements to select
    """
    N = X.size(0)
    CHK = X.split(chunk_size[0])
    D = torch.empty(N, device=X.device)
    I = torch.empty(N, device=X.device)
    size = 0
    for chk in CHK:
        chk_D, chk_I = torch.topk(chk, k[0], dim=-1, largest=bool(largest))
        chk_I = chk_I + size
        D[chk_I] = chk_D
        size += chk.size(0)

    return D, I


@torch.jit.script
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

        if len(kpts) > size:

            # uniform sampling to reduce number of keypoints to desired size
            I = torch.randperm(n_kpts)[:size]
            sampled_kpts = kpts[I]
            KP[idx] = sampled_kpts
            sampled_dtrs = dtrs[I]
            DT[idx] = sampled_dtrs

        else:

            KP[idx, :n_kpts] = kpts
            DT[idx, :n_kpts] = dtrs

            # uniform sampling to fill up to desired size
            diff = size - len(kpts)
            I = torch.arange(n_kpts)

            times = diff // len(kpts)
            I = I.repeat(times + 1)

            I = I[torch.randperm(len(I))][:diff]

            sampled_kpts = kpts[I]
            KP[idx, n_kpts:] = sampled_kpts
            sampled_dtrs = dtrs[I]
            DT[idx, n_kpts:] = sampled_dtrs

    return KP, DT


@torch.jit.script
def match(
    I: torch.Tensor, Q: torch.Tensor, thr: float = 0.7, full_rank_thr: int = 8
) -> torch.Tensor:
    """
    index: (N, K, F) N: number of images, K: number of descriptors, F: descriptor dimension
    query: (N, K, F) N: number of queries, K: number of descriptors, F: descriptor dimension
    thr: ratio test threshold

    idx_idesc_qdesc: (M, 3) M: number of matches, 3: (listwise index, index descriptor, query descriptor)
    """
    # normalize
    sum_ = torch.sum(I)
    sum_ += torch.sum(Q)
    mean_ = sum_ / (torch.numel(I) + torch.numel(Q))
    I = I / mean_
    Q = Q / mean_

    # compute distance matrix of each batch
    # since I is in shape (N, K, F)
    # and Q is in shape  (N, K, F),
    # the distance matrix is in shape (N, K, K)
    # where distance is caluated batchwise
    D = torch.cdist(I, Q)

    # find the top 2 smallest distances for each descriptor
    TD, TI = torch.topk(D, 2, dim=-1, largest=False)
    # TD and TI are both in shape (N, K, 2)

    # ratio test to find matches
    # M is in shape (N, K)
    M = TD[..., 0] < thr * TD[..., 1]

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


@torch.jit.script
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
