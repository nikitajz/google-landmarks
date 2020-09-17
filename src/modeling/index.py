from typing import Union, List
import os
from tqdm import tqdm

import faiss
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def l2norm(x):
    axis = len(x.shape) - 1  # norm each vector on itself
    return x / np.linalg.norm(x, org=2, axis=axis, keepdims=True)


@torch.no_grad()
def extract_features(net, images: Union[List[str], Dataset, DataLoader], device: Union[str, torch.device],
                     normalize=False, save_to_disk=False):
    net.eval()
    net.to(device)
    if isinstance(images, DataLoader):
        loader = images
    else:
        raise NotImplementedError("Not yet implemented")
    meta = []
    vectors_list = []
    for batch in tqdm(loader, desc='Extracting features'):
        meta.append((batch.get('targets', None), batch.get('image_ids', None)))
        features_batch = model.extract_feat(batch['features'].to(device))
        vectors_list.append(features_batch.cpu().numpy().astype('float32'))

    vectors = np.concatenate(vectors_list, axis=0)
    # TODO: norm index
    if normalize:
        raise NotImplementedError()

    if save_to_disk:
        logger.info(f'Saving extracted vectors to checkpoints folder: {CHECKPOINT_DIR}')
        joblib.dump(meta, os.path.join(CHECKPOINT_DIR, 'meta_vectors_train.pkl'))
        joblib.dump(vectors, os.path.join(CHECKPOINT_DIR, 'vectors_train.pkl'))

    return meta, vectors


def build_index(vectors, k=None, ivf=False, path_to_save=None, return_idx=False):
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

    if path_to_save is not None:
        logger.info('Saving index to the file')
        faiss.write_index(faiss_idx, os.path.join(CHECKPOINT_DIR, f"{idx_name}.index"))
    if return_idx:
        return faiss_idx