from typing import Union, List
import os
import logging
from tqdm import tqdm

import faiss
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def l2norm(x):
    axis = len(x.shape) - 1  # norm each vector on itself
    return x / np.linalg.norm(x, ord=2, axis=axis, keepdims=True)


@torch.no_grad()
def extract_features(net: nn.Module,
                     images: Union[List[str], Dataset, DataLoader], device: Union[str, torch.device],
                     mode: str,
                     normalize=False, dir_to_save=None):
    assert mode in ('train', 'test')
    net.eval()
    net.to(device)
    if isinstance(images, DataLoader):
        loader = images
    else:
        raise NotImplementedError("Not yet implemented")
    vectors_list = []
    image_ids = []
    targets = []
    for batch in tqdm(loader, desc='Extracting features'):
        if mode == 'train':
            image_ids.extend(batch['image_ids'])
            targets.append(batch['targets'])
        else:
            image_ids.append(batch['image_ids'])
        features_batch = net.extract_feat(batch['features'].to(device))
        vectors_list.append(features_batch.cpu().numpy().astype('float32'))

    vectors = np.concatenate(vectors_list, axis=0)
    if mode == 'train':
        targets = np.concatenate(targets, axis=0)
    meta = {'image_ids': image_ids,
            'targets': targets}
    # TODO: norm index
    if normalize:
        vectors = l2norm(vectors)

    if dir_to_save is not None:
        logger.info(f'Saving extracted vectors to checkpoints folder: {dir_to_save}')
        joblib.dump(meta, os.path.join(dir_to_save, f'meta_vectors_{mode}.pkl'))
        joblib.dump(vectors, os.path.join(dir_to_save, f'vectors_{mode}.pkl'))

    return meta, vectors


def build_index(vectors, k=None, ivf=False, dir_to_save=None, return_idx=False):
    dim = vectors.shape[1]
    logger.info('Creating FAISS index')
    if ivf:
        quantiser = faiss.IndexFlatL2(dim)
        faiss_idx = faiss.IndexIVFFlat(quantiser, dim, k)
        idx_name = "ivf_flat"
    else:
        if k is not None:
            logger.warning('Using flat index hence `k` is ignored.')
        faiss_idx = faiss.IndexFlatL2(dim)
        idx_name = "flat"

    logger.debug(f'Index is trained: {faiss_idx.is_trained}')  # False.
    logger.debug('Training index')
    faiss_idx.train(vectors)
    logger.info(f'Index training completed: {faiss_idx.is_trained}')  # True

    logger.info('Adding vectors to index')
    logger.debug(f'Initial vectors size: {faiss_idx.ntotal}')  # 0
    faiss_idx.add(vectors)
    logger.info(f'Number of vectors in index: {faiss_idx.ntotal}')

    if dir_to_save is not None:
        logger.info('Saving index to the file')
        faiss.write_index(faiss_idx, os.path.join(dir_to_save, f"{idx_name}.index"))
    if return_idx:
        return faiss_idx
