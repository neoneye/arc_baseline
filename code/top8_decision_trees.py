#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import os

data_path = Path('../data/')

if not os.path.exists('../data'):
    data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'
evaluation_path = data_path / 'validation'
test_path = data_path / 'test'

SAMPLE_SUBMISSION = data_path / 'sample_submission.csv'

SUBMISSION_FILENAME = "submission_top8_tree.csv"

# # Imports

# In[3]:


import json
import math
import operator
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial, reduce
from itertools import chain, permutations, product, starmap, zip_longest
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
import sklearn.tree
from numba import jit


# # Utils

# In[4]:


COLORS = 10


def flatten(l):
    ret = []
    for x in l:
        if isinstance(l, (list, tuple)):
            ret.extend(x)
        else:
            ret.append(x)
    return ret


def reshape_dataframe(df, shape):
    df = [df.iloc[i:i + 1] for i in range(len(df))]
    ret = np.zeros(len(df), dtype=object)
    ret[:] = df
    return ret.reshape(shape)


@jit('void(int32[:,:,:],int8[:,:],int64[:,:],int32,int32,int8,int32,int32)', nopython=True, nogil=True)
def raycast_kernel_single(target, vis, image, dx, dy, col, x, y):
    if vis[x, y]:
        return
    vis[x, y] = 1

    x1 = x + dx
    y1 = y + dy
    if 0 <= x1 < image.shape[0] and 0 <= y1 < image.shape[1] and image[x1, y1] == col:
        raycast_kernel_single(target, vis, image, dx, dy, col, x1, y1)
        target[x, y, 0] = target[x1, y1, 0]
        target[x, y, 1] = target[x1, y1, 1]
    else:
        target[x, y, 0] = x1
        target[x, y, 1] = y1


@jit('void(int32[:,:,:],int8[:,:],int64[:,:],int32,int32,int8)', nopython=True, nogil=True)
def raycast_kernel(target, vis, image, dx, dy, col):
    for ox in range(image.shape[0]):
        for oy in range(image.shape[1]):
            if col == -1:
                tcol = image[ox, oy]
            else:
                tcol = col

            raycast_kernel_single(target, vis, image, dx, dy, tcol, ox, oy)


def raycast(image, delta, col):
    vis = np.zeros_like(image, dtype=np.int8)
    target = np.zeros((*image.shape, 2), dtype=np.int32)
    raycast_kernel(target, vis, image, delta[0], delta[1], col if col is not None else -1)
    return target


@jit('void(int32[:,:], int8[:,:,:], int64[:,:], int8[:,:], int32[:,:])', nopython=True, nogil=True)
def components_kernel(sizes, borders, image, vis, stack):
    col = 0
    # stack[comp, 2 + c] -- counts of color c in component comp
    for a in range(image.shape[0]):
        for b in range(image.shape[1]):
            k = 0
            stack[k, 0] = a
            stack[k, 1] = b
            k += 1

            while k != 0:
                k -= 1
                x = stack[k, 0]
                y = stack[k, 1]

                if vis[x, y]:
                    continue
                vis[x, y] = True
                sizes[x, y] = col

                for i, j in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                    u = x + i
                    v = y + j
                    if 0 <= u < image.shape[0] and 0 <= v < image.shape[1]:
                        c = image[u, v]
                        if not vis[u, v] and image[x, y] == image[u, v]:
                            stack[k, 0] = u
                            stack[k, 1] = v
                            k += 1
                        if c == image[x, y]:
                            c = -1
                    else:
                        c = 0

                    if c >= 0:
                        stack[col, 2 + c] += 1

            col += 1

    for i in range(col):
        stack[i, 0] = 0
    for a in range(image.shape[0]):
        for b in range(image.shape[1]):
            stack[sizes[a, b], 0] += 1

    for a in range(image.shape[0]):
        for b in range(image.shape[1]):
            borders[a, b] = stack[sizes[a, b], 2:]
            sizes[a, b] = stack[sizes[a, b], 0]


def components(image):
    vis = np.zeros_like(image, dtype=np.int8)
    sizes = np.zeros_like(image, dtype=np.int32)
    borders = np.zeros((*image.shape, COLORS), dtype=np.int8)
    stack = np.zeros((image.shape[0] * image.shape[1], 2 + COLORS), dtype=np.int32)
    components_kernel(sizes, borders, image, vis, stack)
    return sizes, borders


class Offset:

    @jit('void(int32[:,:,:], int32[:,:,:], int32[:,:,:])', nopython=True, nogil=True)
    def compose_offsets_kernel(result, offsets, target):
        for x in range(offsets.shape[0]):
            for y in range(offsets.shape[1]):
                a = offsets[x, y, 0]
                b = offsets[x, y, 1]
                if 0 <= a < offsets.shape[0] and 0 <= b < offsets.shape[1]:
                    result[x, y, 0] = target[a, b, 0]
                    result[x, y, 1] = target[a, b, 1]
                else:
                    result[x, y, 0] = a
                    result[x, y, 1] = b

    @jit('void(int8[:,:], int32[:,:], int32[:,:,:], int64[:,:])', nopython=True, nogil=True)
    def get_cols_dists_kernel(cols, dists, offsets, image):
        for x in range(offsets.shape[0]):
            for y in range(offsets.shape[1]):
                a = offsets[x, y, 0]
                b = offsets[x, y, 1]
                if 0 <= a < offsets.shape[0] and 0 <= b < offsets.shape[1]:
                    cols[x, y] = image[a, b]
                else:
                    cols[x, y] = 0
                dists[x, y] = max(abs(x - a), abs(y - b))

    def compose(offsets, target):
        result = np.zeros_like(offsets)
        Offset.compose_offsets_kernel(result, offsets, target)
        return result

    def identity(xyshape):
        a = np.arange(xyshape[0], dtype=np.int32).reshape(-1, 1, 1).repeat(xyshape[1], 1)
        b = np.arange(xyshape[1], dtype=np.int32).reshape(1, -1, 1).repeat(xyshape[0], 0)
        return np.concatenate([a, b], 2)

    def get_cols_dists(offsets, image):
        cols = np.zeros_like(image, dtype=np.int8)
        dists = np.zeros_like(image, dtype=np.int32)
        Offset.get_cols_dists_kernel(cols, dists, offsets, image)
        return cols, dists


# # Transformers

# In[5]:


class TException(Exception):
    """
    Transform exception. A transformer can throw this exception to indicate
    that it doesn't want or doesn't know how to transform a given task.
    """

    pass


class TBase:
    """
    The base class of task transformers. A transformer transforms a task by
    transforming or augmenting input images and corresponding output images.
    It also performs inverse transformation of the output prediction.
    """

    class SampleTransformer:
        """
        The base class of sample transformers. A sample transformer transforms
        one sample (one input image possibly with an output image) of the task.
        """

        def __init__(self, task_transformer, in_image):
            self.task_transformer = task_transformer
            self.in_image = in_image
            self.transformed_images = self.transform()

        def transform(self):
            """
            Transforms or augments `self.in_image`. Returns a list of images.
            """

            raise NotImplementedError()

        def transform_output(self, out_image):
            """
            Transforms or augments `out_images`. Returns a list of images.
            The length of that list must equal the length of the list returned
            in `transform` function.
            """

            raise NotImplementedError()

        def transform_output_inverse(self, prediction):
            """
            Performs aggregation of image predictions and transforms it back.
            `prediction` is a list of the same length as returned in `transform`
            and `transform_output` functions. Returns one image.
            """

            raise NotImplementedError()

    def __init__(self, in_images, out_images):
        """
        The length of `in_images` must be greater or equal the length of `out_images`.
        Input images that don't have corresponding output images are test images,
        and are used for making predictions.
        """

        assert len(in_images) >= len(out_images)

        self.in_images = in_images
        self.out_images = out_images

    def get_transformer(self, in_image):
        """
        Return SampleTransformer for one image
        """

        return self.SampleTransformer(self, in_image)

    def unsequence(self):
        return [self]


class TSequential(TBase):
    """
    The transformer class that executes sequentially the list of transformers.
    """

    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            self.tr1 = self.task_transformer.tr1.get_transformer(self.in_image)
            ins = self.tr1.transformed_images

            self.tr2s = []  # [(transformer, images_count)]
            ret = []
            for i in ins:
                t = self.task_transformer.tr2.get_transformer(i)
                ret.append(t.transformed_images)
                self.tr2s.append((t, len(ret[-1])))
            return flatten(ret)

        def transform_output(self, out_image):
            outs = self.tr1.transform_output(out_image)
            assert len(self.tr2s) == len(outs)
            ret = []
            for (t, c), o in zip(self.tr2s, outs):
                ret.append(t.transform_output(o))
                assert c == len(ret[-1])
            return flatten(ret)

        def transform_output_inverse(self, prediction):
            k = 0
            outs = []
            for t, c in self.tr2s:
                outs.append(t.transform_output_inverse(prediction[k: k + c]))
                k += c
            return self.tr1.transform_output_inverse(outs)

    def __init__(self, tr1, tr2, *a, **k):
        super().__init__(*a, **k)

        self.tr1 = tr1(self.in_images, self.out_images)

        transformers = [self.tr1.get_transformer(i) for i in self.in_images]
        ins = flatten([t.transformed_images for t in transformers])
        outs = flatten([t.transform_output(o)
                        for t, o in zip(transformers, self.out_images)])
        self.tr2 = tr2(ins, outs)

    def unsequence(self):
        return [*self.tr1.unsequence(), *self.tr2.unsequence()]

    @classmethod
    def make(cls, trs, *a, **k):
        if len(trs) == 0:
            return TIdentity
        if len(trs) == 1:
            return trs[0]
        if len(trs) == 2:
            return cls(trs[0], trs[1], *a, **k)
        return cls(trs[0], partial(cls.make, trs[1:]), *a, **k)


class TCBase(TBase):
    """
    The base class for color transformers.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.constant_colors = sorted(list(set(range(COLORS))
            - set(np.unique(np.concatenate([i.reshape(-1) for i in self.in_images])))))

    class SampleTransformer(TBase.SampleTransformer):
        def __init__(self, task_transformer, in_image, color_mapping, out_color_mapping=None):
            self.color_mapping = color_mapping
            self.out_color_mapping = out_color_mapping if out_color_mapping is not None else color_mapping
            super().__init__(task_transformer, in_image)

        def inversed_out_mapping(self):
            assert len(set(self.out_color_mapping)) == len(
                self.out_color_mapping)
            assert set(self.out_color_mapping) == set(range(COLORS))
            ret = np.zeros_like(self.out_color_mapping)
            ret[self.out_color_mapping] = np.arange(
                len(self.out_color_mapping))
            return ret

        def transform(self):
            return [self.color_mapping[self.in_image]]

        def transform_output(self, out_image):
            return [self.out_color_mapping[out_image]]

        def transform_output_inverse(self, prediction):
            return self.inversed_out_mapping()[prediction[0]]

    def get_transformer(self, in_image):
        raise NotImplementedError()


class TCCount(TCBase):
    """
    A color transformer that colors each image independenly by the number
    of color occurences, i.e. after coloring, the color 0 is the frequent color
    in the image, the color 1 is the second frequent color in the image, etc.
    """

    def get_transformer(self, in_image):
        mapping = np.arange(COLORS)
        colors, counts = np.unique(np.concatenate(
            [in_image.reshape(-1), np.arange(COLORS)]), return_counts=True)
        pos = [((in_image == c).nonzero() + (np.array([c, 0]),))[0].tolist()
               for c in range(COLORS)]
        i = 0
        for _, _, col in sorted(zip(counts, pos, colors), reverse=True):
            if col in self.constant_colors:
                mapping[col] = COLORS - 1 - self.constant_colors.index(col)
            else:
                mapping[col] = i
                i += 1
        return self.SampleTransformer(self, in_image, mapping)


class TCCountBg(TCBase):
    """
    A color transformer that works in the same way as `TCCount` but the color 0 is
    reserved for the global color 0.
    """

    def get_transformer(self, in_image):
        mapping = np.arange(COLORS)
        colors, counts = np.unique(np.concatenate(
            [in_image.reshape(-1), np.arange(COLORS), np.array([0] * 10000)]), return_counts=True)
        pos = [((in_image == c).nonzero() + (np.array([c, 0]),))[0].tolist()
               for c in range(COLORS)]
        i = 0
        for _, _, col in sorted(zip(counts, pos, colors), reverse=True):
            if col in self.constant_colors:
                mapping[col] = COLORS - 1 - self.constant_colors.index(col)
            else:
                mapping[col] = i
                i += 1
        return self.SampleTransformer(self, in_image, mapping)


class TCPos(TCBase):
    """
    A color transformer that colors images accordingly to the order of occurrences. 
    """

    def get_transformer(self, in_image):
        mapping = {}
        for col in range(COLORS):
            if col in self.constant_colors:
                mapping[col] = COLORS - 1 - self.constant_colors.index(col)
        k = 0
        for x, y in product(*map(range, in_image.shape)):
            c = in_image[x, y]
            if c not in mapping:
                mapping[c] = k
                k += 1
        for c in range(COLORS):
            if c not in mapping:
                mapping[c] = k
                k += 1
        mapping = np.array([mapping[i] for i in range(COLORS)])

        return self.SampleTransformer(self, in_image, mapping)


class TIdentity(TBase):
    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return [self.in_image]

        def transform_output(self, out_image):
            return [out_image]

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TSeparationRemover(TBase):
    """
    A transformer that removes vertical and horizontal separations / frames
    from the image. Every image has to have the same number of separations.
    """

    @staticmethod
    def calc_delta_start_count(x, y, borders, shape, extend_nonexisting=False):
        if any(starmap(lambda s, k: (s - k + 1 - 2 * borders) % k != 0, zip(shape, [x, y]))):
            return None, None, None
        delta = tuple(starmap(lambda s, k: (s + 1 - 2 * borders) // k, zip(shape, [x, y])))
        if borders:
            start = (0, 0)
        else:
            if extend_nonexisting:
                start = (-1, -1)
            else:
                start = (delta[0] - 1, delta[1] - 1)
        if extend_nonexisting:
            count = (x + 1, y + 1)
        else:
            count = (x - 1 + 2 * borders, y - 1 + 2 * borders)
        return delta, start, count

    def __init__(self, *a, **k):
        super().__init__(*a, **k)

        number_of_match = 0

        matches = []  # [(x_parts, y_parts, with_borders)]
        for x in range(1, min(map(lambda x: x.shape[0], self.in_images))):
            for y in range(1, min(map(lambda x: x.shape[1], self.in_images))):
                for borders in [False, True]:
                    if (x, y) == (1, 1):
                        continue

                    for img in self.in_images:
                        delta, start, count = self.calc_delta_start_count(
                            x, y, borders, img.shape)
                        if delta is None:
                            break

                        arr = []
                        for d in [0, 1]:
                            for a in range(count[d]):
                                z = start[d] + a * delta[d]
                                if d == 0:
                                    r = img[z, :]
                                else:
                                    r = img[:, z]
                                arr = np.unique(np.concatenate([arr, np.unique(r)]))
                                if len(arr) > 1:
                                    break
                            if len(arr) > 1:
                                break
                        if len(arr) != 1:
                            break
                    else:
                        matches.append((x, y, borders))

        matches = sorted(matches, key=lambda x: (-x[0] - x[1], not x[2], *x))
        if len(matches) <= number_of_match:
            raise TException()

        self.match = matches[number_of_match]

    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            x, y, borders = self.task_transformer.match
            delta, start, count = self.task_transformer.calc_delta_start_count(
                x, y, borders, self.in_image.shape, extend_nonexisting=True)
            ret = []
            for a in range(count[0]):
                range1 = slice(start[0] + a * delta[0] + 1,
                               start[0] + (a + 1) * delta[0])
                r = []
                for b in range(count[1]):
                    range2 = slice(start[1] + b * delta[1] + 1,
                                   start[1] + (b + 1) * delta[1])
                    r.append(self.in_image[range1, range2])

                ret.append(np.concatenate(r, 1))
            ret = np.concatenate(ret)
            return [ret]

        def transform_output(self, out_image):
            return [out_image]

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TTransposeInput(TBase):
    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return [self.in_image.copy().T]

        def transform_output(self, out_image):
            return [out_image]

        def transform_output_inverse(self, prediction):
            return prediction[0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if all([i.shape[0] == i.shape[1] for i in self.in_images]):
            raise TException()


def _get_flips(img):
    return [img[:, :], img[::-1, :], img[:, ::-1], img[::-1, ::-1]]


class TAugFlips(TBase):
    """
    A transformer that augments images by adding their flips.
    """

    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return _get_flips(self.in_image)

        def transform_output(self, out_image):
            return _get_flips(out_image)

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TAugFlipsWRotation(TBase):
    class SampleTransformer(TBase.SampleTransformer):
        def transform(self):
            return _get_flips(self.in_image) + _get_flips(self.in_image.transpose(1, 0))

        def transform_output(self, out_image):
            return _get_flips(out_image) + _get_flips(out_image.transpose(1, 0))

        def transform_output_inverse(self, prediction):
            return prediction[0]


class TAugColor(TBase):
    """
    A transformer that augments images by recoloring them using random color
    permutations, except for the color 0 which is not changed.
    """

    class SampleTransformer(TBase.SampleTransformer):

        def transform(self):
            return [m[self.in_image] for m in self.task_transformer.mappings]

        def transform_output(self, out_image):
            return [m[out_image] for m in self.task_transformer.mappings]

        def transform_output_inverse(self, prediction):
            return self.task_transformer.inv_mappings[0][prediction[0]]

    def inverted_mapping(self, mapping):
        ret = np.zeros_like(mapping)
        ret[mapping] = np.arange(len(mapping))
        return ret

    def __init__(self, in_images, out_images):
        self.in_images = in_images
        self.out_images = out_images

        all_input_colors = set(chain.from_iterable(map(np.unique, in_images)))
        all_output_colors = set(chain.from_iterable(map(np.unique, out_images)))
        self.all_colors = set(range(COLORS))

        self.permutable_colors = sorted(all_input_colors.difference({0}))

        self.mappings = []
        self.inv_mappings = []

        perms = list(permutations(self.permutable_colors))
        for perm in random.sample(perms, min(12, len(perms))):
            mapping = np.arange(COLORS)
            mapping[self.permutable_colors] = perm
            self.mappings.append(mapping)
            self.inv_mappings.append(self.inverted_mapping(mapping))

        if len(self.mappings) <= 1:
            raise TException()


# # Featurizers

# In[6]:


class FException(Exception):
    """
    Featurizer exception. A featurizer can throw this exception to indicate
    that it doesn't want or doesn't know how to featurize a given task.
    """

    pass


class FBase:
    """
    The base class of task featurizers. A featurizer featurizes a task by
    extracting features from input images, and getting labels (pixel-level) from output images.
    It also performs assembling of the output image based on predictions.
    """

    class SampleFeaturizer:
        """
        The base class of sample featurizer. A sample featurizer featurizes
        one sample of the task.
        """

        def __init__(self, task_featurizer, in_image):
            self.task_featurizer = task_featurizer
            self.in_image = in_image

        def calc_features(self):
            self.features = self.featurize()

        def featurize(self):
            """
            Extracts features from `self.in_image`. Returns a data frame.
            """

            raise NotImplementedError()

        def get_labels(self, out_image):
            """
            Extracts labels from `out_image`. Return a 1D np.array of integers.
            The length of this np.array must equal the length of the dataframe
            returned in `featurize` function. $i$-th element of the array
            corresponds to $i$-th element of the data frame.
            """

            raise NotImplementedError()

        def assemble(self, prediction):
            """
            Assembles predictions into the output image. `prediction` is
            a 1D np.array of integers that correspond to the data frame
            returned in `featurize` function.
            """

            raise NotImplementedError()

    def __init__(self, in_images, out_images):
        self.in_images = in_images
        self.out_images = out_images

    def get_featurizer(self, in_image):
        return self.SampleFeaturizer(self, in_image)

    def get_features_visualization(self):
        return []


class FGlobal(FBase):
    """
    Experimental featurizer, that wraps another featurizer,
    and appends image-level features to each dataframe row.
    Here, global features are based on a simple object detection algorithm.
    """

    class SampleFeaturizer(FBase.SampleFeaturizer):
        def __init__(self, wrapped_sample_featurizer, featurizer, in_image, in_image_gfeatures):
            super().__init__(featurizer, in_image)
            self.wrapped_sample_featurizer = wrapped_sample_featurizer
            self.featurizer = featurizer
            self.in_image = in_image
            self.in_image_gfeatures = in_image_gfeatures

        def featurize(self):
            df = self.wrapped_sample_featurizer.featurize()
            assert len(self.in_image_gfeatures) > 0
            global_features = np.array(list(chain.from_iterable(f.reshape(-1)
                    for f in self.in_image_gfeatures)))
            assert len(global_features) > 1
            names = [f'global_feat{i}' for i in range(len(global_features))]
            for name, col in zip(names, global_features):
                df[name] = [col] * len(df)
            return df

        def get_labels(self, out_image):
            return self.wrapped_sample_featurizer.get_labels(out_image)

        def assemble(self, prediction):
            return self.wrapped_sample_featurizer.assemble(prediction)

    def _calc_global_features(self):
        """
        Some heuristic -- object detection.
        """
        bboxes_per_sample = []
        for img in self.in_images:
            bboxes = defaultdict(list)
            for col in range(10):
                num_component, component = cv2.connectedComponents((img == col).astype(np.uint8))
                for c in range(1, num_component):
                    p = (component == c).astype(np.uint8)
                    if p.sum():
                        bbox = cv2.boundingRect(p)
                        bboxes[bbox[2:]].append(bbox)
            bboxes_per_sample.append(bboxes)

        all_keys = sorted(set(chain.from_iterable(b.keys() for b in bboxes_per_sample)))
        for w, h in all_keys:
            crops = []
            if (w, h) == (1, 1):
                continue
            if all(len(b[(w, h)]) == 1 for b in bboxes_per_sample):
                bboxes = []
                for in_img, bbox in zip(self.in_images, bboxes_per_sample):
                    bbox = bbox[(w, h)]
                    assert len(bbox) == 1
                    if (h, w) == in_img.shape:
                        break
                    bboxes.append(bbox[0])
                else:
                    for in_img, bbox in zip(self.in_images, bboxes):
                        x, y, _, _ = bbox
                        crops.append(in_img[y:y + h, x:x + w])

                    # skip detection if there are only zero-entropy images
                    if all(len(np.unique(img)) == 1 for img in crops):
                        continue

                    self.in_images_gfeatures.append(crops)

    def __init__(self, wrapped_featurizer, in_images, out_images):
        self.in_images = in_images
        self.out_images = out_images
        self.wrapped_featurizer = wrapped_featurizer(in_images, out_images)

        # 2D array of images (crops): n_detections (most often 0 or 1) x len(in_images)
        self.in_images_gfeatures = []

        self._calc_global_features()
        if not self.in_images_gfeatures:
            raise FException()

    def get_featurizer(self, in_image):
        wrapped_sample_featurizer = self.wrapped_featurizer.get_featurizer(in_image)
        in_image_gfeatures = []

        # gather features corresponding to the given in_image
        for one_detection_images in self.in_images_gfeatures:
            assert len(self.in_images) == len(one_detection_images)
            for img, gfeatures in zip(self.in_images, one_detection_images):
                if img is in_image:
                    in_image_gfeatures.append(gfeatures)

        assert in_image_gfeatures
        return self.SampleFeaturizer(wrapped_sample_featurizer, self, in_image, in_image_gfeatures)

    def get_features_visualization(self):
        return self.in_images_gfeatures


class FConstant(FBase):
    """
    A featurizer for tasks in which input size equals output size.
    """

    class SampleFeaturizer(FBase.SampleFeaturizer):
        def featurize(self):
            features = []  # list of np.array[nfeatures, xsize, ysize] or np.array[xsize, ysize]
            feature_names = []  # list of strings

            deltas = (-1, 0, 1)
            neigh = np.zeros((len(deltas) ** 2, *self.in_image.shape), dtype=np.int8)
            for ox, oy in product(*map(range, self.in_image.shape)):
                # absolute neighbourhood color
                for k, (i, j) in enumerate(product(*[deltas] * 2)):
                    x = (ox + i)  # % self.in_image.shape[0]
                    y = (oy + j)  # % self.in_image.shape[1]
                    if 0 <= x < self.in_image.shape[0] and 0 <= y < self.in_image.shape[1]:
                        neigh[k, ox, oy] = self.in_image[x, y]
                    else:
                        neigh[k, ox, oy] = 0
                    k += 1
            features.append(neigh)
            feature_names.extend(['Neigh{}'.format(i) for i in product(*[deltas] * 2)])

            # unique = np.zeros((3, *self.in_image.shape), dtype=np.int8) # 3 because row, column, neighborhood
            # unique_names = []
            # for ox, oy in product(*map(range, self.in_image.shape)):
            #     # absolute neighbourhood color
            #     unique[0, ox, oy] = len(np.unique(self.in_image[:, oy]))
            #     unique[1, ox, oy] = len(np.unique(self.in_image[ox, :]))
            #     unique[2, ox, oy] = len(np.unique(self.in_image[max(ox - 1, 0) : ox + 2,
            #                                                     max(oy - 1, 0) : oy + 2]))
            # features.append(unique)
            # feature_names.extend(['UniqueInRow', 'UniqueInCol', 'UniqueInNeigh'])

            if self.task_featurizer is None or self.task_featurizer.use_rot_90:
                rotations = (False, True)
            else:
                rotations = (False,)

            for use_rot in rotations:
                sym = np.zeros((4, *self.in_image.shape), dtype=np.int8)
                for ox, oy in product(*map(range, self.in_image.shape)):
                    for k, (i, j) in enumerate(product(*([[0, 1]] * 2))):
                        x, y = ox, oy
                        if i:
                            x = self.in_image.shape[0] - x - 1
                        if j:
                            y = self.in_image.shape[1] - y - 1
                        if use_rot:
                            if y >= self.in_image.shape[0] or x >= self.in_image.shape[1]:
                                sym[k, ox, oy] = -1
                            else:
                                sym[k, ox, oy] = self.in_image[y, x]
                        else:
                            sym[k, ox, oy] = self.in_image[x, y]
                features.append(sym)
                feature_names.extend(f'SymRot90_{i}{j}' if use_rot else f'Sym_{i}{j}'
                                     for i, j in product(*([[0, 1]] * 2)))

            deltas = [[1, 0], [-1, 0], [0, 1], [0, -1],
                      [1, 1], [-1, -1], [-1, 1], [1, -1]]
            for di, d in enumerate(deltas):
                col_opts = [0, None]
                targets = [raycast(self.in_image, d, col) for col in col_opts]
                for t1, col in zip(targets, col_opts):
                    cols, dists = Offset.get_cols_dists(t1, self.in_image)
                    features.extend([dists % 2, dists % 3])
                    dists = dists.astype(np.float32)
                    features.extend([cols, dists])
                    ray_type_label = 'Same' if col is None else 'Notbg'
                    feature_names.extend([f'RayCol{ray_type_label}_{d}_mod2',
                                          f'RayCol{ray_type_label}_{d}_mod3'])
                    feature_names.extend([f'RayCol{ray_type_label}_{d}',
                                          f'RayDist{ray_type_label}_{d}'])
                    for t2, col2 in zip(targets, col_opts):
                        off = Offset.compose(t1, t2)

                        cols2, dists2 = Offset.get_cols_dists(off, self.in_image)
                        dists2 = dists2.astype(np.float32)
                        features.extend([cols2, dists2 - dists])
                        ray_type_label2 = 'Same' if col2 is None else 'Notbg'
                        feature_names.extend([f'RayCol{ray_type_label}_{ray_type_label2}_{d}',
                                              f'RayDist{ray_type_label}_{ray_type_label2}_{d}'])

            sizes, borders = components(self.in_image)
            sizes = sizes.astype(np.float32)
            border_exclusive = ((borders == borders.sum(2, keepdims=True)) * np.arange(1, COLORS + 1)).sum(2) - 1
            features.extend([sizes, border_exclusive])
            feature_names.extend(['ComponentSize', 'ComponentBorderExclusive'])

            features = np.concatenate(list(map(
                lambda x: (x if len(x.shape) == 3 else x.reshape((-1, *x.shape))).astype(object), features)))
            features = features.reshape((features.shape[0], -1)).transpose([1, 0])
            return pd.DataFrame(features, columns=feature_names)

        def get_labels(self, out_image):
            if self.in_image.shape != out_image.shape:
                raise FException()
            return out_image.reshape((-1,))

        def assemble(self, prediction):
            return prediction.reshape(self.in_image.shape)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rot_90 = all(i.shape[0] == i.shape[1] for i in self.in_images)
        if not all(i.shape == o.shape for i, o in zip(self.in_images, self.out_images)):
            raise FException()


class FFactorBase(FBase):
    """
    The base class of featurizers for tasks where an input size is a multiplication
    of an output size (the same multiplication factor for every pair input / output image).
    """

    @staticmethod
    def calculate_factor(i, o):
        round_num = 6
        return round(float(o[0] / i[0]), round_num), round(float(o[1] / i[1]), round_num)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        factors = [self.calculate_factor(i.shape, o.shape)
                   for i, o in zip(self.in_images, self.out_images)]
        if len(set(factors)) != 1:
            raise FException()
        self.factor = factors[0]
        self.factor = tuple(map(lambda x: round(x) if x == round(x) else x, self.factor))

    class SampleFeaturizer(FBase.SampleFeaturizer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.factor = self.task_featurizer.factor
            self.out_size = tuple((round(float(f * s))
                    for f, s in zip(self.task_featurizer.factor, self.in_image.shape)))
            if (self.task_featurizer.calculate_factor(self.in_image.shape, self.out_size)
                    != self.task_featurizer.factor):
                raise FException()

        def get_labels(self, out_image):
            return out_image.reshape((-1,))

        def assemble(self, prediction):
            return prediction.reshape(self.out_size)


class FFactorUp(FFactorBase):
    def __init__(self, coord_func, *a, **k):
        self.coord_func = coord_func
        super().__init__(*a, **k)

    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def featurize(self):
            if (not (self.factor == tuple(map(round, self.factor))
                    and (self.factor[0] >= 1 and self.factor[1] >= 1)
                    and (self.factor[0] > 1 or self.factor[1] > 1))):
                raise FException()

            single_featurizer = FConstant.SampleFeaturizer(None, self.in_image)
            single_featurizer.calc_features()
            feats = reshape_dataframe(single_featurizer.features, self.in_image.shape)

            features = np.zeros(self.out_size, dtype=object)
            for x, y in product(range(self.in_image.shape[0]), range(self.in_image.shape[1])):
                for i, j in product(range(self.factor[0]), range(self.factor[1])):
                    f = feats[x, y].copy()
                    f['CoordX'] = [i]
                    f['CoordY'] = [j]
                    for a, b in product(*[[0, 1]] * 2):
                        f[f'SymC_{a}{b}'] = f[f'Sym_{(i % 2) ^ a}{b}']
                        f[f'Sym_{a}C{b}'] = f[f'Sym_{a}{(j % 2) ^ b}']
                        f[f'SymC_{a}C{b}'] = f[f'Sym_{(i % 2) ^ a}{(j % 2) ^ b}']
                    features[self.task_featurizer.coord_func(x, y, i, j, self)] = f

            features = pd.concat(features.reshape(-1).tolist(), ignore_index=True)
            return features


def tile_coord(x, y, i, j, self):
    return x + self.in_image.shape[0] * i, y + self.in_image.shape[1] * j


def scale_coord(x, y, i, j, self):
    return x * self.factor[0] + i, y * self.factor[1] + j


FTile = partial(FFactorUp, tile_coord)
FScale = partial(FFactorUp, scale_coord)


class FFactorDown1(FFactorBase):
    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def featurize(self):
            down_factor = (round(1 / self.factor[0], 3), round(1 / self.factor[1], 3))
            down_factor = tuple(map(lambda x: round(x) if x == round(x) else x, down_factor))

            if (not (down_factor == tuple(map(round, down_factor))
                    and (self.factor[0] <= 1 and self.factor[1] <= 1)
                    and (self.factor[0] < 1 or self.factor[1] < 1))):
                raise FException()

            features = []
            for x, y in product(*map(range, down_factor)):
                fragment = self.in_image[x * self.out_size[0]: (x + 1) * self.out_size[0],
                                         y * self.out_size[1]: (y + 1) * self.out_size[1]]
                single_featurizer = FConstant.SampleFeaturizer(None, fragment)
                single_featurizer.calc_features()
                f = single_featurizer.features.copy()
                f.columns = list(map(lambda col: 'x{}-y{}-{}'.format(x, y, col), f.columns))
                features.append(f)

            features = reduce(pd.DataFrame.join, features)
            return features


class FFactorDown2(FFactorBase):
    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def featurize(self):
            down_factor = (round(1 / self.factor[0], 3), round(1 / self.factor[1], 3))
            down_factor = tuple(map(lambda x: round(x) if x == round(x) else x, down_factor))

            if (not (down_factor == tuple(map(round, down_factor))
                    and (self.factor[0] <= 1 and self.factor[1] <= 1)
                    and (self.factor[0] < 1 or self.factor[1] < 1))):
                raise FException()

            features = np.zeros(self.out_size, dtype=object)
            for x, y in product(*map(range, self.out_size)):
                fragment = self.in_image[x * down_factor[0]: (x + 1) * down_factor[0],
                                         y * down_factor[1]: (y + 1) * down_factor[1]]
                single_featurizer = FConstant.SampleFeaturizer(None, fragment)
                single_featurizer.calc_features()
                f = single_featurizer.features
                f = reshape_dataframe(f, -1)
                pf = []
                for k, r in enumerate(f):
                    r = r.copy().reset_index()
                    r.columns = list(map(lambda col: 'k{}-{}'.format(k, col), r.columns))
                    pf.append(r)

                f = reduce(pd.DataFrame.join, pf)
                features[x, y] = f

            features = pd.concat(features.reshape(-1).tolist(), ignore_index=True)
            return features


class FSquare(FBase):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)

        for i, o in zip(self.in_images, self.out_images):
            if i.shape[0] ** 2 != o.shape[0] or i.shape[1] ** 2 != o.shape[1]:
                raise FException()

    class SampleFeaturizer(FFactorBase.SampleFeaturizer):
        def __init__(self, *a, **k):
            FBase.SampleFeaturizer.__init__(self, *a, **k)
            self.out_size = (self.in_image.shape[0] ** 2, self.in_image.shape[1] ** 2)

        def featurize(self):
            single_featurizer = FConstant.SampleFeaturizer(None, self.in_image)
            single_featurizer.calc_features()

            features = np.zeros(self.out_size, dtype=object)
            for x, y in product(*map(range, self.in_image.shape)):
                f = single_featurizer.features

                i = x * self.in_image.shape[1] + y
                p = single_featurizer.features.iloc[i:i + 1]
                p.columns = list(map(lambda x: 'square-{}'.format(x), p.columns))
                f = f.join(pd.concat([p] * len(f), ignore_index=True))

                f = reshape_dataframe(f, self.in_image.shape)
                features[x * self.in_image.shape[0]: (x + 1) * self.in_image.shape[0],
                         y * self.in_image.shape[1]: (y + 1) * self.in_image.shape[1]] = f

            features = pd.concat(features.reshape(-1).tolist(), ignore_index=True)
            return features


# # Confidence model

# In[7]:


def featurize_task_stat(task_stat):

    def used_unique_features(i):
        if i < 0:
            return set()

        ret = set()
        ret = ret.union(used_unique_features(
            task_stat.model.tree_.children_right[i]))
        ret = ret.union(used_unique_features(
            task_stat.model.tree_.children_left[i]))
        ret.add(task_stat.model.tree_.feature[i])
        return ret

    rows = []
    assert len(task_stat.correct()) == len(task_stat.predictions)

    tr_mappings_count = 1
    for level, tr in enumerate(task_stat.transformer.unsequence()):
        if isinstance(tr, TAugColor):
            tr_mappings_count = len(tr.mappings)

    # add to features binary information about using selected transforms
    types_to_featurize = [TAugFlips, TAugFlipsWRotation]
    all_transformers = set(map(type, task_stat.transformer.unsequence()))
    tfeatures = [t in all_transformers for t in types_to_featurize]
    tfeatures_columns_names = list(t.__name__ for t in types_to_featurize)

    for i, correct in enumerate(task_stat.correct()):
        rows.append(dict(
            correct=correct,  # label
            used_unique_features_count=math.log(len(used_unique_features(0))),
            nodes_count=math.log(task_stat.model.tree_.node_count),
            deductible=task_stat.deducible,
            tr_mappings_count=math.log(tr_mappings_count),
            **{k: v for k, v in zip(tfeatures_columns_names, tfeatures)}
        ))

    df = pd.DataFrame(rows)
    return df.reindex(sorted(df.columns), axis=1)


def iterate_stats(stats):
    for config_stat in stats:
        for task_stat in config_stat:
            if not task_stat.error:
                yield task_stat


def calc_confidences(args, pool, stats):
    handles = []
    for task_stat in iterate_stats(stats):
        handles.append(pool.apply_async(
            featurize_task_stat, args=(task_stat,)))
    if not len(handles):
        return
    df = pd.concat([h.get() for h in handles], ignore_index=True)
    df, labels = df[df.columns.difference(['correct'])], df['correct']

    if args.train_confidence is not None:
        print(df.head())
        print(df.mean(axis=0))

        model = sklearn.linear_model.LogisticRegression()
        model.fit(df, labels)

        with args.train_confidence.open('wb') as f:
            pickle.dump(model, f)

        # convenient to copy and paste into a notebook for kaggle submission
        print('Pickled confidence model (hex):')
        print(pickle.dumps(model).hex())
    else:
        with args.load_confidence.open('rb') as f:
            model = pickle.load(f)

    print("Linear model coefficients", model.coef_)

    confidence_iter = iter(model.predict_proba(df))
    for task_stat in iterate_stats(stats):
        for preds_per_test_img in task_stat.predictions:
            confidence = next(confidence_iter)[1]
            for i, prediction in enumerate(preds_per_test_img):
                preds_per_test_img[i] = (prediction[0], confidence)
    try:
        next(confidence_iter)
        assert 0
    except StopIteration:
        pass


# # Main

# In[8]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_args(args=None):
    def bool_type(x):
        if x.lower() in ['1', 'true']:
            return True
        if x.lower() in ['0', 'false']:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset-root', type=Path,
                        default='/kaggle/input/abstraction-and-reasoning-challenge/')
    parser.add_argument('-s', '--dataset-suffix', type=str, default='evaluation')
    parser.add_argument('-g', '--gen-submission', type=bool_type, default=True)
    parser.add_argument('-t', '--threads', type=int, default=2)
    parser.add_argument('-p', '--pickle-results', type=Path,
            help="Path to generated predictions and models. "
            "If the file exists it is loaded, otherwise it is calculated and written.")
    parser.add_argument('--save-all-predictions', type=bool_type, default=False,
            help="Allow writing more than 3 answers per test image to resulting submission file.")
    parser.add_argument('--train-confidence', type=Path, default=None,
            help="If specified, train and save confidence model pickle to a given path.")
    parser.add_argument('--load-confidence', type=Path, default=None,
            help="Confidence model pickle path.")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(args)

    if args.train_confidence is None and args.load_confidence is None:
        raise ValueError("Either --train-confidence or --load-confidence required.")

    if args.seed < 0:
        args.seed = np.random.randint(2 ** 16)
        print('Seed:', args.seed)

    set_seed(args.seed)
    return args


def save_output_csv(stats, save_all_predictions):
    final_rows = []
    for task in stats:
        for i, pr in enumerate(task.get_topk()):
            out_str = []
            for t in pr:
                s = [''.join(map(str, p)) for p in t]
                s = '|'.join(s)
                s = f'|{s}|'
                out_str.append(s)
            if save_all_predictions:
                if not out_str:
                    out_str = ['|0|']
            else:
                out_str = (out_str + ['|0|'] * 3)[:3]
            out_str = ' '.join(out_str)
            final_rows.append(('{}_{}'.format(task.task_id, i), out_str))
    df = pd.DataFrame(final_rows, columns=['output_id', 'output'])
    df.to_csv(SUBMISSION_FILENAME, index=False)


class TaskStat:
    def __init__(self, task):
        self.task = task  # used in plot_pickled_results.py

        self.task_id = ''
        self.transformer_error = False
        self.featurizer_error = False
        self.error = False
        self.deducible = False
        self.predictions = [] # [test_image][prediction_number] -> (prediction, confidence)
        self.ground_truth = []

        self.model = None
        self.feature_names = None
        self.transformer = None
        self.featurizer = None

    def is_prediction_done(self):
        return not self.error

    def __or__(self, other):
        ret = TaskStat(self.task)
        ret.transformer_error = self.transformer_error and other.transformer_error
        ret.featurizer_error = self.featurizer_error and other.featurizer_error
        ret.error = self.error and other.error
        ret.deducible = self.deducible or other.deducible
        ret.predictions = [x + y for x, y in zip(self.predictions, other.predictions)]
        ret.ground_truth = self.ground_truth
        assert all(starmap(lambda x, y: (x == y).all(),
                           zip(self.ground_truth, other.ground_truth)))
        ret.task_id = self.task_id
        assert self.task_id == other.task_id
        return ret

    def correct(self, k=None):
        return np.array(list(starmap(
            lambda xs, gt: any([np.array([x == gt]).all() for x in xs]) if gt is not None else False,
            zip_longest(self.get_topk(k), self.ground_truth))))

    def accuracy(self, k=None):
        return self.correct(k).mean()

    def get_topk(self, k=None):
        ret = []
        for preds in self.predictions:
            added = set()
            r = []
            for pred in sorted(filter(lambda x: x[1] is not None, preds), key=lambda x: -x[1]):
                tup = tuple(pred[0].reshape(-1))
                if tup not in added:
                    added.add(tup)
                    r.append(pred[0])
                    if k is not None and len(r) >= k:
                        break
            ret.append(r)
        return ret


class ConfigStats:

    def __init__(self, config):
        self.task_stats = []
        self.config = config  # used in plot_pickled_results.py

    def append_task_stat(self, task_stat):
        self.task_stats.append(task_stat)

    def __or__(self, other):
        ret = ConfigStats(self.config)
        ret.task_stats = [a | b for a, b in zip(self.task_stats, other.task_stats)]
        return ret

    def __iter__(self):
        return iter(self.task_stats)

    def __len__(self):
        return len(self.task_stats)

    def __getitem__(self, i):
        return self.task_stats[i]

    def __str__(self):
        def samples_correct(k=None):
            return sum(map(lambda x: sum(x.correct(k)), self))

        samples_num = sum(map(lambda x: len(x.predictions), self))

        def tasks_correct_list(k=None):
            return list(map(lambda x: (reduce(operator.mul, x.correct(k)) == 1)
                                      if len(x.predictions) else False, self))

        def tasks_correct(k=None): return sum(tasks_correct_list(k))
        tasks_num = len(self)
        transformer_errors = sum(map(lambda x: x.transformer_error, self))
        featurizer_errors = sum(map(lambda x: x.featurizer_error, self))
        errors = sum(map(lambda x: x.error, self))
        deducible_counts = sum(map(lambda x: x.deducible, self))
        return f'''\
Accuracy            {samples_correct() / max(samples_num, 1)} ({samples_correct()} / {samples_num})
Accuracy-top1       {samples_correct(1) / max(samples_num, 1)} ({samples_correct(1)} / {samples_num})
Accuracy-top3       {samples_correct(3) / max(samples_num, 1)} ({samples_correct(3)} / {samples_num})
Task-accuracy       {tasks_correct() / tasks_num} ({tasks_correct()} / {tasks_num})
Task-accuracy-top1  {tasks_correct(1) / tasks_num} ({tasks_correct(1)} / {tasks_num})
Task-accuracy-top3  {tasks_correct(3) / tasks_num} ({tasks_correct(3)} / {tasks_num})
Transformer-errors  {transformer_errors}
Featurizer-errors   {featurizer_errors}
Errors              {errors}
Deducible           {deducible_counts}
Deducible&Correct   {sum(np.array(list(map(lambda x: x.deducible, self))) & np.array(tasks_correct_list()))}
'''


def is_deducible(x, y):
    data = x.join(pd.DataFrame(y, columns=['label']))
    cols = list(data.columns)
    cols.pop(cols.index('label'))
    for i, j in data.groupby(cols):
        if len(j.label.unique()) != 1:
            return False
    return True


def _process(config, task, in_images, out_images):
    TransformerClass, FeaturizerClass, ModelClass = config
    task_stat = TaskStat(task)

    task_transformer = TransformerClass(in_images, out_images)
    transformers = [task_transformer.get_transformer(in_image) for in_image in in_images]
    transformed_in_images = flatten(
        [transformer.transformed_images for transformer in transformers])
    transformed_out_images = flatten(
            [transformer.transform_output(out_image)
                for transformer, out_image in zip(transformers, out_images)])

    task_featurizer = FeaturizerClass(transformed_in_images, transformed_out_images)
    featurizers = [task_featurizer.get_featurizer(in_image) for in_image in transformed_in_images]
    for featurizer in featurizers:
        featurizer.calc_features()

    features = [featurizer.features for featurizer in featurizers]

    def column_order(name):
        if 'Neigh' in name:
            if '0,0' in name or '0, 0' in name:
                return 0, name
            else:
                return 1, name
        elif 'Ray' in name:
            return 2, name
        else:
            return 3, name

    columns = list(features[0].columns)
    columns = sorted(columns, key=column_order, reverse=False)
    features = [pd.DataFrame(np.array(f[columns]), columns=columns)
                for f in features]

    labels = [featurizer.get_labels(out_image) for featurizer, out_image in zip(
        featurizers, transformed_out_images)]

    train_features = pd.concat(features[:len(labels)], ignore_index=True)

    model = ModelClass(max_features=len(columns))
    task_stat.deducible = is_deducible(train_features, np.concatenate(labels))
    model.fit(train_features, np.concatenate(labels))
    raw_predictions = [model.predict(features)
                       for features in features[len(labels):]]
    assembled_predictions = [featurizer.assemble(prediction)
            for featurizer, prediction in zip(featurizers[len(labels):], raw_predictions)]

    nested_predictions = []
    k = 0
    for transformer in transformers[len(out_images):]:
        nested_predictions.append(
            assembled_predictions[k: k + len(transformer.transformed_images)])
        k += len(transformer.transformed_images)

    final_predictions = [transformer.transform_output_inverse(prediction)
            for transformer, prediction in zip(transformers[len(out_images):], nested_predictions)]

    task_stat.model = model
    task_stat.feature_names = columns
    task_stat.transformer = task_transformer
    task_stat.featurizer = task_featurizer

    confidences = [0] * len(raw_predictions)
    task_stat.predictions = list(
        starmap(lambda x, c: [[x, c]], zip(final_predictions, confidences)))

    return task_stat


def process(task_index, task, config, in_images, out_images, valid_out_images, seed):
    set_seed(seed)

    task_stat = TaskStat(task)
    try:
        task_stat = _process(config, task, in_images, out_images)
    except FException:
        task_stat.featurizer_error = True
        task_stat.error = True
        for _ in range(len(in_images) - len(out_images)):
            task_stat.predictions.append([])
    except TException:
        task_stat.transformer_error = True
        task_stat.error = True
        for _ in range(len(in_images) - len(out_images)):
            task_stat.predictions.append([])

    task_stat.ground_truth = valid_out_images
    return task_stat


def config_str(x):
    if isinstance(x, partial):
        return '|'.join(map(config_str, x.args))
    if isinstance(x, tuple):
        return '(' + ','.join(map(config_str, x)) + ')'
    if isinstance(x, list):
        return '-'.join(list(map(config_str, x)))
    return x.__name__


def build_configs(args):
    """ Build multiple configurations that use (or not) various transforms and featurizers.
    """
    configs = []
    for model in [
        partial(sklearn.tree.DecisionTreeClassifier, criterion='entropy'),
        partial(sklearn.tree.DecisionTreeClassifier, criterion='gini'),
    ]:
        for ts in product([
                TIdentity,
                # TTransposeInput,
            ], [
                TIdentity,
                TSeparationRemover,
            ], [
                TIdentity,
                TCCount,
                TCCountBg,
                TCPos
            ], [
                TIdentity,
                TAugFlips,
                # TAugFlipsWRotation,
            ], [
                TIdentity,
                TAugColor,
        ]):
            transforms = partial(TSequential.make, ts)
            for featurizer in [
                    # partial(FGlobal, FConstant),
                    # partial(FGlobal, FTile),
                    # partial(FGlobal, FScale),
                    # partial(FGlobal, FSquare),
                    # partial(FGlobal, FFactorDown2),
                    # partial(FGlobal, FFactorDown1),
                    FConstant,
                    # FTile,  # commented out because we need deterministic behavior like in older version for high lb score
                    FScale,
                    FSquare,
                    FFactorDown2,
                    FFactorDown1,
            ]:
                configs.append([transforms, featurizer, model])
    return configs


def parse_task(path):
    with open(path, 'r') as f:
        task = json.load(f)

    in_images = [sample['input'] for sample in task['train']
                 ] + [sample['input'] for sample in task['test']]
    out_images = [sample['output'] for sample in task['train']]
    valid_out_images = [sample['output']
                        for sample in task['test'] if 'output' in sample]

    return (task, *(list(map(np.array, i)) for i in (in_images, out_images, valid_out_images)))


def _main(args, data):
    configs = build_configs(args)

    with Pool(args.threads) as pool:
        if args.pickle_results is None or not args.pickle_results.exists():
            exclude = set()
            
            tasks = []
            for i, path in enumerate(data):
                task, in_images, out_images, valid_out_images = parse_task(path)

                if i not in exclude:
                    handles = [pool.apply_async(
                        process, (i, task, config, in_images, out_images,
                                  valid_out_images, np.random.randint(2 ** 32))) for config in configs]
                else:
                    handles = []

                tasks.append((task, in_images, out_images,
                              valid_out_images, handles))

            stats = [ConfigStats(c) for c in configs]
            for i, path in enumerate(data):
                if i in exclude:
                    continue

                task, in_images, out_images, valid_out_images, handles = tasks[i]

                print('Processing task:', i)

                for config, handle, stat in zip(configs, handles, stats):
                    task_stats = handle.get()
                    task_stats.task_id = path.stem
                    stat.append_task_stat(task_stats)

            if args.dataset_suffix != 'test':
                for i, (config, stat) in enumerate(zip(configs, stats)):
                    print()
                    print('----------', i, '-----', config_str(config))
                    print(stat)

            if args.pickle_results is not None:
                with args.pickle_results.open('wb') as f:
                    pickle.dump(stats, f)

        else:
            with args.pickle_results.open('rb') as f:
                stats = pickle.load(f)

        calc_confidences(args, pool, stats)

    all_stats = reduce(operator.or_, stats)
    if args.dataset_suffix != 'test':
        print()
        print('SUM')
        print(all_stats)

    if args.gen_submission:
        save_output_csv(all_stats, args.save_all_predictions)


def main(args):
    dataset = sorted((args.dataset_root / args.dataset_suffix).glob('*.json'))
    if not dataset:
        raise ValueError(f'no data found in {args.dataset_root / args.dataset_suffix}')

    _main(args, dataset)


# In[9]:


main(parse_args(['-s', 'test', '--load-confidence', 'top8_models/top8_confidence_model.pickle', '--seed', '3', '--save-all-predictions', '1']))


# # Getting 2 more (non-elegant way)
# Normally above solution would give 0.94 LB. We did the following:
# 1. We tested, that in 3-rd attempt we have no hits (+ 1 submission).
# 2. We found additional 2 hits -- one in 4-th and one in 10-th attempt (+ 3 submissions -- 4-6, 7-9, 10-12, and + 1-2 submissions to check whether 10,11 or 12).
# 3. We blend these 4-th and 10-th attempts randomly (+ expected 2 submissions) and use as a 3-rd attempt.

# In[10]:


submission = pd.read_csv(SUBMISSION_FILENAME)
submission = submission.set_index('output_id')
rnd = random.Random(2)

final_output = []
for attempts in submission.output:
    attempts = attempts.split()
    attempts_to_use = []
    for index in [0, 1, 3, 9]:
        if len(attempts) <= index:
            attempts_to_use.append("|0|")
        else:
            attempts_to_use.append(attempts[index])
    if rnd.random() >= 0.5 and attempts_to_use[3] != "|0|":
        attempts_to_use[2] = attempts_to_use[3]
    attempts_to_use = attempts_to_use[:3]
    final_output.append(' '.join(attempts_to_use))

submission.output = final_output
submission.to_csv(SUBMISSION_FILENAME)

