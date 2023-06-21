
from pathlib import Path
import os


data_path = Path('../data/')

if not os.path.exists('../data'):
    data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'
evaluation_path = data_path / 'validation'
test_path = data_path / 'test'

SAMPLE_SUBMISSION = data_path / 'sample_submission.csv'


SUBMISSION_FILENAME = "submission_crop5.csv"

# ----------------------------------------


import numpy as np
import os
import pandas as pd
import json

import sys
import copy
import json
import os
import pandas as pd
import numpy as np


# simple operations that change numpy array
from collections import Counter
from functools import partial
import itertools

import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes


paths = {'train': training_path, 'eval': evaluation_path, 'test': test_path}

"""
from arclib.dsl import *
from arclib.util import evaluate_predict_func, test_path, get_string, data_path
from arclib.dsl import Task, unique_arrays
from arclib.check import  check_output_color_from_input
"""

# ---------- dsl: BEGIN ----------
# simple operations that change numpy array
from collections import Counter
from functools import partial
import itertools

import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes


### Operation applied to numpy array

def pad(array, size=1, color=0):
    # size in (1, 10), color: int
    shape = array.shape
    new_array = np.full((shape[0] + 2 * size, shape[1] + 2 * size), color)
    new_array[size:-size, size:-size] = array
    return new_array


def recolor(array, old_color, new_color):
    array_ = array.copy()
    array_[array_ == old_color] = new_color
    return array_


def crop_without_bg(array, bg_color):
    # crop rectangle with content
    array_ = array.copy()
    mask = (array_ != bg_color).astype(int)
    args = np.argwhere(mask > 0)
    axis0_min = np.min(args, axis=0)[0]
    axis1_min = np.min(args, axis=0)[1]
    axis0_max = np.max(args, axis=0)[0]
    axis1_max = np.max(args, axis=0)[1]
    array_ = array_[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]
    return array_


def repeat(array, n0, n1):
    # n0 in range(2, 10)
    # n1 in range(2, 10)
    return np.tile(array, (n0, n1))


def rotate(array, angle=1):
    # angle in (1,2,3)
    return np.rot90(array, angle)


def flip(array, axis=0):
    # axis in (0, 1)
    return np.flip(array, axis)


def flip_diagonal(array, axis=0):
    # axis in (0, 1)
    if axis == 0:
        return np.swapaxes(array, 0, 1)
    elif axis == 1:
        return np.rot90(np.flip(array, 0), 1)


def repeat_map(array, map):
    # map consists from 0 and 1
    k = array.shape[0]
    l = array.shape[1]
    shape = (map.shape[0] * k, map.shape[1] * l)
    array_ = np.zeros(shape)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 1:
                array_[i * k : (i + 1) * k, j * l : (j + 1) * l] = array
    return array_


def zoom(array, n0, n1):
    array_ = np.kron(array, np.ones((n0, n1)))
    return array_


def identity(array):
    return array


### reversed functions


def reverse_zoom(array, n0, n1):
    shape = array.shape
    k0 = shape[0] // n0
    k1 = shape[1] // n1
    array_ = np.empty((k0, k1))
    for i in range(k0):
        for j in range(k1):
            array_[i, j] = array[i * n0, j * n1]
    return array_


def reverse_repeat(array, n0, n1):
    s0, s1 = array.shape
    k0 = s0 // n0
    k1 = s1// n1
    return array[:k0, : k1]


#### Brute force

range10 = tuple(range(1, 11))
range4 = tuple(range(1, 5))
simple_funcs = [identity, rotate, flip, flip_diagonal, zoom, repeat]
reversed_simple_funcs = [identity, rotate, flip, flip_diagonal, reverse_zoom, reverse_repeat]
simple_func_param_dicts = [{}, {'angle': (1,2,3)}, {'axis': (0,1)}, {'axis': (0, 1)}] + [{'n0': range4, 'n1': range4}] + [{'n0': range4, 'n1': range4}]
reversed_simple_func_param_dicts = [{}, {'angle': (-1,-2,-3)}, {'axis': (0,1)}, {'axis': (0, 1)}] + [{'n0': range10, 'n1': range10}] + [{'n0': range10, 'n1': range10}]


def generate_param_options(param_dict):
    keys = list(param_dict.keys())
    param_values = list(param_dict.values())
    param_options =  list(itertools.product(*param_values))
    param_option_dicts = [{key: value for key, value in zip(keys, param_option)} for param_option in param_options]
    return param_option_dicts


def get_partials(funcs, param_dicts):
    options = []
    for func, param_dict in zip(funcs, param_dicts):
        param_option_dicts = generate_param_options(param_dict)
        func_options = [partial(func, **param_option_dict) for param_option_dict in param_option_dicts]
        options.extend(func_options)
    return options


simple_output_process_options = get_partials(simple_funcs, simple_func_param_dicts)
reversed_simple_output_process_options = get_partials(reversed_simple_funcs, reversed_simple_func_param_dicts)

### Operation detectors

# use output only
def detect_true(array):
    return True


def get_divisors(n):
    divisors = []
    for i in range(1, n//2 + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors + [n]


def detect_zoom(array, n0, n1):
    s0, s1 = array.shape
    axis0_divisors = get_divisors(array.shape[0])
    axis1_divisors = get_divisors(array.shape[1])
    if n0 not in axis0_divisors:
        return False
    if n1 not in axis1_divisors:
        return False
    if n0 == 1 and n1 == 1:
        return False
    if n0 == array.shape[0] and n1 == array.shape[1]:
        return False
    subarrays = get_equal_shape_subarrays(array, n0, n1)
    for subarray in subarrays:
        if len(all_colors(subarray)) != 1:
            return False
    return True


def detect_repeat(array, n0, n1):
    s0, s1 = array.shape
    axis0_divisors = get_divisors(s0)
    axis1_divisors = get_divisors(s1)
    if n0 not in axis0_divisors:
        return False
    if n1 not in axis1_divisors:
        return False
    if n0 == 1 and n1 == 1:
        return False
    if n0 == s0 and n1 == s1:
        return False
    subarrays = get_equal_shape_subarrays(array, s0 // n0, s1 // n1)
    if len(unique_arrays(subarrays)) != 1:
        return False
    return True

# while doing brute force of simple functions run operation only if detect func returns True
detect_simple_functions = [detect_true] * 8 + get_partials([detect_zoom], [{'n0': range10, 'n1': range10}]) + get_partials([detect_repeat], [{'n0': range10, 'n1': range10}])



### SKIMAGE OBJECTS

def get_objects_from_map_(object_map, touch='wall'):
    if touch == 'wall':
        connectivity = 1
    elif touch == 'corner':
        connectivity = 2
    label_array, max_label_index = label(object_map, connectivity=connectivity,
                                         background=0, return_num=True)
    return [(label_array == i).astype(int) for i in range(1, max_label_index + 1)]


def get_most_common_array_color(array):
    colors, color_counts = np.unique(array, return_counts=True)
    most_common_color = colors[np.argmax(color_counts)]
    return most_common_color


def touchs_boundary(map):
    sums = [np.count_nonzero(map[0, 1:-1]),  np.count_nonzero(map[-1, 1:-1]), np.count_nonzero(map[1:-1, 0]), np.count_nonzero(map[1:-1, -1])]
    return sum([s > 0 for s in sums]) >= 3


def detect_bg_(array, how='touch'):
    # TODO: not sure that  i need detect_bg
    if how == 'touch':
        bg = 0
        colors = all_colors(array)
        most_common_color = get_most_common_array_color(array)
        for color in [most_common_color] + list(colors - {most_common_color}):
            object_maps = get_objects_from_map_((array == color).astype(int), touch='corner')
            for object_map in object_maps:
                if touchs_boundary(object_map):
                    bg = color
                    break
    elif how == 'black':
        bg = 0
    elif how == 'most_common':
        bg = get_most_common_array_color(array)
    return bg


def get_objects_by_connectivity_(array, touch='wall', bg_color=None):
    if bg_color is None:
        bg_color = detect_bg_(array)
    array_map = (array != bg_color).astype(int)
    object_maps = get_objects_from_map_(array_map, touch=touch)
    return object_maps


def get_objects_by_color_and_connectivity_(array, touch='wall', bg_color=None):
    if bg_color is None:
        bg_color = detect_bg_(array)
    colors = all_colors(array) - {bg_color}
    object_maps = []
    for color in colors:
        color_object_maps = get_objects_from_map_((array == color).astype(int), touch=touch)
        object_maps.extend(color_object_maps)
    return object_maps


def get_objects_by_color_(array, bg_color=None):
    if bg_color is None:
        bg_color = detect_bg_(array)
    object_maps = []
    colors = all_colors(array) - {bg_color}
    for color in colors:
        object_map = (array == color).astype(int)
        object_maps.append(object_map)
    return object_maps



def get_objects_rectangles(array, direction='vertical', bg_color=None):
    # rectangles of same color can be attached to each other #
    s0, s1 = array.shape
    if bg_color is None:
        bg_color = detect_bg_(array)
    object_maps = []
    #colors = all_colors(array) - {bg_color}
    if direction == 'horisontal':
        object_map = np.zeros_like(array)
        prev_slice = np.full((s1,), -100)
        for i in range(s0):
            slice = array[i]
            if (np.unique(slice) != np.array(bg_color)).any():
                if (slice != prev_slice).any():
                    if np.sum(object_map) != 0:
                        object_maps.append(object_map)
                        object_map = np.zeros_like(array)
                object_map[i] = slice
            else:
                if np.sum(object_map) != 0:
                    object_maps.append(object_map)
                    object_map = np.zeros_like(array)
            prev_slice = slice
        if np.sum(object_map) != 0:
            object_maps.append(object_map)
    elif direction == 'vertical':
        object_map = np.zeros_like(array)
        prev_slice = np.full((s0,), -100)
        for i in range(s1):
            slice = array[:, i]
            if (np.unique(slice) != np.array(bg_color)).any():
                if (slice != prev_slice).any():
                    if np.sum(object_map) != 0:
                        object_maps.append(object_map)
                        object_map = np.zeros_like(array)
                object_map[:, i] = slice
            else:
                if np.sum(object_map) != 0:
                    object_maps.append(object_map)
                    object_map = np.zeros_like(array)
            prev_slice = slice
        if np.sum(object_map) != 0:
            object_maps.append(object_map)
    object_maps = [(object_map > 0).astype(int) for object_map in object_maps]
    return object_maps


def fit_largest_rectangle(object_map):
    largest_rectangle_map = np.zeros_like(object_map)

    nrows, ncols = object_map.shape
    skip = 0
    area_max = (0, [])

    a = object_map
    w = np.zeros(dtype=int, shape=a.shape)
    h = np.zeros(dtype=int, shape=a.shape)
    for r in range(nrows):
        for c in range(ncols):
            if a[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, [(r - dh, c - minw + 1, r, c)])
    axis0_min, axis1_min, axis0_max, axis1_max = area_max[1][0]
    largest_rectangle_map[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = 1
    return largest_rectangle_map


def remove_padding(object_map):
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    object_map_ = np.zeros_like(object_map)
    object_map_[axis0_min + 1: axis0_max, axis1_min + 1: axis1_max] = 1
    return object_map_


def get_objects_rectangles_without_noise(array, bg_color=None):
    # get rectangle objects out of noisy background
    object_maps = get_objects_by_color_and_connectivity_(array, touch='wall', bg_color=bg_color)
    object_maps = [object_map for object_map in object_maps if np.count_nonzero(object_map) > 1]
    object_maps = [binary_fill_holes(object_map).astype(int) for object_map in object_maps]
    object_maps = [fit_largest_rectangle(object_map) for object_map in object_maps]
    return object_maps


def get_objects_rectangles_without_noise_without_padding(array, bg_color=None):
    object_maps = get_objects_rectangles_without_noise(array, bg_color=bg_color)
    object_maps = [remove_padding(object_map) for object_map in object_maps]
    return object_maps


def get_equal_shape_subarrays(array, l0, l1):
    # (l0, l1) - shape of subarrays
    # untile operations but parameters are dimensions of new array
    subarrays = []
    s0, s1 = array.shape
    n0 = s0 // l0
    n1 = s1 // l1
    for i in range(n0):
        for j in range(n1):
            subarray = array[i * l0: (i + 1) * l0, j * l1 : (j + 1) * l1]
            subarrays.append(subarray)
    return subarrays


def get_object_parts(object_map):
    shape = object_map.shape
    border_map = np.zeros_like(object_map)
    for i in range(shape[0]):
        subarray = object_map[i]
        args = np.argwhere(subarray == 1)[:, 0]
        if len(args) > 0:
            args_min = np.min(args)
            args_max = np.max(args)
            border_map[i, args_min] = 1
            border_map[i, args_max] = 1

    for j in range(shape[1]):
        subarray = object_map[:, j]
        args = np.argwhere(subarray == 1)[:, 0]
        if len(args) > 0:
            args_min = np.min(args)
            args_max = np.max(args)
            border_map[args_min, j] = 1
            border_map[args_max, j] = 1
    inner_map = (object_map != border_map).astype(int)
    return border_map, inner_map


### Object mask operations


def get_rectangle_map(object_map):
    rectangle_mask = np.zeros_like(object_map)
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    rectangle_mask[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = 1
    return rectangle_mask


def get_cropped_map(object_map):
    cropped_mask = crop_without_bg(object_map, bg_color=0)
    return cropped_mask


def get_colored_map(array, object_map):
    colored_map = array.copy()
    colored_map[object_map == 0] = -1
    return colored_map


### Object features

def get_object_map_min_max(object_map):
    args = np.argwhere(object_map > 0)
    axis0_min = np.min(args, axis=0)[0]
    axis1_min = np.min(args, axis=0)[1]
    axis0_max = np.max(args, axis=0)[0]
    axis1_max = np.max(args, axis=0)[1]
    return axis0_min, axis0_max, axis1_min, axis1_max


def get_object_dimensions(object_map):
    cropped_mask = get_cropped_map(object_map)
    return cropped_mask.shape


def get_object_size(object_map):
    return np.sum(object_map)


def get_most_common_color(array, object_map):
    colors, color_counts = np.unique(array[object_map == 1], return_counts=True)
    most_common_color = colors[np.argmax(color_counts)]
    return most_common_color


def get_sorted_colors(array, object_map=None):
    if object_map is not None:
        array = array[object_map == 1]
    colors, color_counts = np.unique(array, return_counts=True)
    return colors(np.argsort(color_counts))


def get_object_colors_cnt(array, object_map):
    return Counter(array[object_map == 1].tolist())


def is_rectangle(object_map):
    rectangle_mask = get_rectangle_map(object_map)
    if (rectangle_mask == object_map).all():
        return True
    return False


### Object class


class BaseObject:
    #  object without nested objects
    def __init__(self, array, object_map):
        self.array = array
        self.object_map = object_map
        self.rectangle_map = get_rectangle_map(object_map)
        self.colored_map = get_colored_map(array, object_map)
        self.cropped_map = get_cropped_map(object_map)
        self.cropped_object = crop_without_bg(self.colored_map, bg_color=-1) # bg_color=-1
        self.size = get_object_size(object_map)
        self.height, self.width = self.cropped_map.shape
        self.area = self.height * self.width
        self.is_rectangle = is_rectangle(object_map)
        self.colors, self.color_counts = np.unique(array[object_map == 1], return_counts=True)
        self.n_colors = len(self.colors)
        #self.unique_colors = set(self.colors)
        self.most_common_color = self.colors[np.argmax(self.color_counts)]
        self.least_common_color = self.colors[np.argmin(self.color_counts)]
        self.axis0_min, self.axis0_max, self.axis1_min, self.axis1_max = get_object_map_min_max(self.object_map)


exclude_feature_names = ['array', 'nested_object', 'nested_objects', 'color_counts', 'object_map', 'colored_map']
min_max_feature_names = ['size', 'height', 'width', 'area', 'n_colors', 'axis0_min', 'axis0_max', 'axis1_min', 'axis1_max',
                         'nested_object_size', 'nested_objects_count']

feature_names = min_max_feature_names + ['cropped_map', 'cropped_object', 'is_rectangle',
                                         'most_common_color', 'least_common_color',
                                         'nested_object_shape', 'nested_object_most_common_color']

class Object(BaseObject):
    # object with nested objects
    def __init__(self, array, object_map):
        super().__init__(array, object_map)
        self.nested_objects = make_base_objects(array, get_nested_objects(array, object_map))
        self.nested_objects_count = len(self.nested_objects)
        if len(set(np.unique(self.colored_map)) - {-1}) > 1: # nested object exists
            self.nested_object = BaseObject(array, get_nested_object(array, object_map, self.most_common_color))
            self.nested_object_size = self.nested_object.size
            self.nested_object_shape = self.nested_object.cropped_map
            self.nested_object_most_common_color = self.nested_object.most_common_color
        else:
            self.nested_object = None
            self.nested_object_size = None
            self.nested_object_shape = None
            self.nested_object_most_common_color = None

    def feature_dict(self):
        # all object features in one dictionary
        feature_dict_ = {}
        for attr, value in self.__dict__.items():
            if attr in feature_names:
                if type(value) == np.ndarray:
                    value_ = array_to_tuple(value)
                else:
                    value_ = value
                feature_dict_[attr] = value_
        return feature_dict_


def make_objects(array, object_maps):
    return [Object(array, object_map) for object_map in object_maps]


def make_base_objects(array, object_maps):
    return [BaseObject(array, object_map) for object_map in object_maps]


def get_nested_objects(array, object_map):
    # get all nested objects separately
    array_ = array.copy()
    array_[object_map == 0] = -1
    nested_object_maps = get_objects_by_color_and_connectivity_(array_, touch='wall', bg_color=-1)
    return nested_object_maps


def get_nested_object(array, object_map, most_common_color):
    # get all nested objects all together even if they are not connected
    # exclude most common color
    array_ = array.copy()
    array_[object_map == 0] = -1
    array_[array_ == most_common_color] = -1
    array_[array_ >= 0] = 1
    array_[array_ == -1] = 0
    nested_object_map = array_ #get_objects_by_color_(array_, bg_color=-1)[0]
    return nested_object_map


# Task class and color functions
class Task():
    def __init__(self, task, idx=None):
        self.task = task
        self.inputs = get_inputs(task)
        self.outputs = get_outputs(task)
        self.test_inputs = get_test_inputs(task)
        self.test_outputs = get_test_outputs(task)
        self.test_pairs = get_test_pairs(task)
        self.pairs = get_pairs(task)
        self.idx = idx


def flatten_list(pred):
    return [p for row in pred for p in row]


def get_outputs(task):
    train_examples = task['train']
    outputs = [ex['output'] for ex in train_examples]
    outputs = [np.array(output) for output in outputs]
    return outputs


def get_inputs(task):
    train_examples = task['train']
    inputs = [ex['input'] for ex in train_examples]
    inputs = [np.array(input) for input in inputs]
    return inputs


def get_pairs(task):
    train_examples = task['train']
    pairs = [(np.array(ex['input']), np.array(ex['output'])) for ex in train_examples]
    return pairs


def get_test_inputs(task):
    test = task['test']
    return [np.array(ex['input']) for ex in test]


def get_test_outputs(task):
    test = task['test']
    if 'output' in test[0].keys():
        return [np.array(ex['output']) for ex in test]
    else:
        return None


def get_test_pairs(task):
    test = task['test']
    if 'output' in test[0].keys():
        return [(np.array(ex['input']), np.array(ex['output'])) for ex in test]
    else:
        return None


def colors(array):
    colors = set(flatten_list(array)) - {0}
    return colors


def all_colors(array):
    colors = set(flatten_list(array))
    return colors


def array_to_tuple(array):
    if len(array.shape) == 2:
        res = tuple(tuple(a.tolist()) for a in array)
    else:
        res = tuple(array.tolist())
    return res


def unique_arrays(arrays):
    arrays = [array_to_tuple(a) for a in arrays]
    arrays = list(set(arrays))
    arrays = [np.array(a) for a in arrays]
    return arrays


def all_task_colors(task):
    arrays = task.inputs + task.outputs + task.test_inputs
    colors = set(flatten_list([all_colors(array) for array in arrays]))
    return colors


def colors_only_in_task_outputs(task):
    input_colors = set(flatten_list([all_colors(array) for array in task.test_inputs]))
    output_colors = set(flatten_list([all_colors(array) for array in task.test_outputs]))
    diff = output_colors - input_colors
    return diff


def colors_only_in_input(input_, output):
    input_colors = all_colors(input_)
    output_colors = all_colors(output)
    return input_colors - output_colors


def colors_only_in_output(input_, output):
    input_colors = all_colors(input_)
    output_colors = all_colors(output)
    return output_colors - input_colors
# ---------- dsl: END ----------

# ---------- util: BEGIN ----------
import copy
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path



def evaluate_predict_func(func):
    train_tasks = get_tasks('train')
    eval_tasks = get_tasks('eval')
    print('Evaluating train')
    train_score, train_string = evaluate_func_on_tasks(func, train_tasks)
    print('Evaluating eval')
    eval_score, eval_string = evaluate_func_on_tasks(func, eval_tasks)

    print('Train score:', train_score, ';', train_string)
    print('Eval score:', eval_score, ';', eval_string)

    return train_score , eval_score


def evaluate_func_on_tasks(func, tasks):
    preds = [func(task) for task in tasks]
    score, string = score_predictions(tasks, preds)
    return score, string


def score_predictions(tasks, predictions):
    n = 0
    tp = 0
    for task, task_preds in zip(tasks, predictions):
        task_tp = score_task_predictions(task, task_preds)
        tp += task_tp
        n += len(task.test_pairs)
        if task_tp > 0:
            print('True prediction:', task.idx)
    return (n - tp)/float(n), str(tp) + '/' + str(n)


def score_task_predictions(task, task_preds, data='test'):
    if data == 'test':
        examples = task.test_pairs
    elif data == 'train':
        examples = task.pairs
    else:
        raise Exception("Data parameter can be 'train' or 'test' ")

    tp = 0
    for example, example_preds in zip(examples, task_preds):
        example_score = score_example_predictions(example, example_preds)
        tp += example_score
    return tp


def score_example_predictions(example, example_preds):
    input_, output = example
    true = output
    if type(example_preds) == np.ndarray:
        example_preds = [example_preds]
    for pred in example_preds:
        if pred.shape == true.shape:
            if (pred == true).all():
                return 1
    return 0


def get_string(pred):
    pred = pred.astype(int)
    str_pred = str([list(row) for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def get_tasks(dataset='train'):
    path = paths[dataset]
    fns = sorted(os.listdir(path))
    tasks = []
    for idx, fn in enumerate(fns):
        fp = path / fn
        with open(fp, 'r') as f:
            task = Task(json.load(f), idx)
            tasks.append(task)
    return tasks


def sets_intersection(sets_list):
    x = sets_list[0]
    for s in sets_list[1:]:
        x = x.intersection(s)
    return x


def sets_union(sets_list):
    x = set()
    for s in sets_list:
        x = x.union(s)
    return x


def arrays_equal(outputs, predicted_outputs):
    for out, pred in zip(outputs, predicted_outputs):
        if (out != pred).any():
            return False
    return True


def change_inputs(task, inputs_, data='train'):
    task_ = copy.deepcopy(task)
    if data == 'train':
        task_.inputs = inputs_
        task_.pairs = [(inp, out) for inp, out in zip(inputs_, task_.outputs)]
    elif data == 'test':
        task_.test_inputs = inputs_
    return task_


def change_outputs(task, outputs_, data='train'):
    task_ = copy.deepcopy(task)
    if data == 'train':
        task_.outputs = outputs_
        task_.pairs = [(inp, out) for inp, out in zip(task_.inputs, task_.outputs)]
    elif data == 'test':
        task_.test_outputs = outputs_
    return task_


def change_inputs_outputs(task, inputs_, outputs_, data='train'):
    if data == 'train':
        task_ = copy.deepcopy(task)
        task_.pairs = [(inp, out) for inp, out in zip(inputs_, outputs_)]
        task_.inputs = inputs_
        task_.outputs = outputs_
    return task_


def apply_array_func(task, array_func, where='both', data='both'):
    # applys function to all task train inputs and train outputs
    if where == 'both':
        inputs_ = [array_func(input_) for input_ in task.inputs]
        outputs_ = [array_func(output) for output in task.outputs]
        task_ = change_inputs_outputs(task, inputs_, outputs_)
        test_inputs_ = [array_func(input_) for input_ in task.test_inputs]
        if task.test_outputs[0]:
            test_outputs_ = [array_func(output) for output in task.test_outputs]
        else:
            test_outputs_ = task.test_outputs
        task_ = change_inputs_outputs(task_, test_inputs_, test_outputs_, data='test')
    elif where == 'inputs':
        inputs_ = [array_func(input_) for input_ in task.inputs]
        task_ = change_inputs(task, inputs_)
        test_inputs_ = [array_func(input_) for input_ in task.test_inputs]
        task_ = change_inputs(task_, test_inputs_,  data='test')
    elif where == 'outputs':
        outputs_ = [array_func(output) for output in task.outputs]
        task_ = change_outputs(task, outputs_)

    return task_

# ---------- util: END ----------

# ---------- check: BEGIN ----------
def check_outputs_one_color(task):
    one_color = True
    for output in task.outputs:
        if len(set(flatten_list(output))) != 1:
            one_color = False
    return one_color


def check_outputs_same_shape(task):
    shapes = []
    for output in task.outputs:
        shapes.append(output.shape)
    return len(set(shapes)) == 1


def check_output_color_from_input(task):
    for pair in task.pairs:
        input_, output = pair
        input_colors = all_colors(input_)
        output_colors = all_colors(output)
        if output_colors - input_colors != set():
            return False
    return True


def check_input_output_same_shape(task):
    outputs = task.outputs
    inputs = task.inputs

    same_shape = True
    for inp, out in zip(inputs, outputs):
        if inp.shape != out.shape:
            same_shape = False
    return same_shape


def check_outputs_smaller(task):
    if task.idx == 66:
        a = 1
    for input, output in task.pairs:
        if input.shape[0] < output.shape[0] or  input.shape[1] < output.shape[1]:
            return False
        if input.shape[0] == output.shape[0] and  input.shape[1] <= output.shape[1]:
            return False
        if input.shape[0] < output.shape[0] and  input.shape[1] == output.shape[1]:
            return False
    return True
# ---------- check: END ----------


# ---------- 5-crop-tasks-by-brute-force.ipynb: BEGIN ----------

def check_output_in_candidates(output, candidates):
    output_is_candidate = False

    for candidate in candidates:
        if output.shape == candidate.shape:
            if (output == candidate).all():
                output_is_candidate = True
    return output_is_candidate


def predict_part(task, get_candidates, train_object_maps=None, train_bg_colors=None):

    part_task = True
    for i, (input, output) in enumerate(task.pairs):
        input = np.array(input)
        output = np.array(output)

        candidates = get_candidates(input, object_maps=train_object_maps[i], bg_color=train_bg_colors[i])

        if candidates:
            if check_output_in_candidates(output, candidates) == False:
                part_task = False
                break
        else:
            part_task = False
            break

    all_input_predictions = []
    if part_task:
        for input in task.test_inputs:
            test_candidates = get_candidates(input)
            predictions = test_candidates #[:3]
            predictions = unique_arrays(predictions)
            predictions = sorted(predictions, key= lambda x: x.shape[0] * x.shape[1], reverse=True)
            all_input_predictions.append(predictions)

    else:
        all_input_predictions = []
    return all_input_predictions


def get_cropped_object(array, object_map, bg_color=None):
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    return array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]


def keep_one_object(array, object_map, bg_color=None):
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    if bg_color is None:
        bg_color = detect_bg_(array)
    output_ = np.full_like(array, bg_color)
    output_[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]
    return output_


def get_cropped_objects(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):
    if object_maps is None:
        object_maps = get_object_maps(array)
    objects = [augment(get_cropped_object(array, object_map)) for object_map in object_maps if np.count_nonzero(object_map) > 0]
    return objects


def get_inputs_with_one_object(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):
    if object_maps is None:
        object_maps = get_object_maps(array)
    objects = [augment(keep_one_object(array, object_map, bg_color=bg_color)) for object_map in object_maps if np.count_nonzero(object_map) > 0]
    return objects


get_object_map_funcs = [ get_objects_by_connectivity_, partial(get_objects_by_connectivity_, touch='corner'),
                    get_objects_by_color_and_connectivity_, partial(get_objects_by_color_and_connectivity_, touch='corner'), get_objects_by_color_,
                    get_objects_rectangles, partial(get_objects_rectangles, direction='horisontal'), get_objects_rectangles_without_noise, get_objects_rectangles_without_noise_without_padding]

def predict_part_types(task):

    predictions = []

    if check_output_color_from_input(task):

        bg_colors = [detect_bg_(input_) for input_ in task.inputs]
        for i, get_object_maps in enumerate(get_object_map_funcs):

            object_maps_list = [get_object_maps(input_, bg_color=bg_color) for input_, bg_color in zip(task.inputs, bg_colors)]
            for get_object_func in (get_cropped_objects, get_inputs_with_one_object):
                for augment in simple_output_process_options:
                    get_candidates = partial(get_object_func, get_object_maps=get_object_maps, augment=augment)
                    predictions = predict_part(task, get_candidates=get_candidates, train_object_maps=object_maps_list, train_bg_colors=bg_colors)
                    if predictions:
                        break
                if predictions:
                    break
            if predictions:
                break


    return predictions


def submit(predict):
    submission = pd.read_csv(SAMPLE_SUBMISSION, index_col='output_id')
    submission['output'] = ''
    test_fns = sorted(os.listdir(test_path))

    print(len(test_fns))
    print(test_fns[:10])
    print(test_fns[-10:])

    count = 0
    for fn in test_fns:
        fp = test_path / fn
        with open(fp, 'r') as f:
            task = Task(json.load(f))
            all_input_preds = predict(task)
            if all_input_preds:
                print(fn)
                count += 1

            for i, preds in enumerate(all_input_preds):
                output_id = str(fn.split('.')[-2]) + '_' + str(i)
                string_preds = [get_string(pred) for pred in preds[:3]]
                pred = ' '.join(string_preds)
                submission.loc[output_id, 'output'] = pred

    print(count)
    submission.to_csv(SUBMISSION_FILENAME)


def main():
    #evaluate_predict_func(predict_part_types)
    submit(predict_part_types)


if __name__ == '__main__':
    main()
