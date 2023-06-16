
from pathlib import Path
import os

data_path = Path('../data/')

if not os.path.exists('../data'):
    data_path = Path('/data/')

training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'evaluation'

SAMPLE_SUBMISSION = str(data_path / "solution" / 'sample_submission.csv')

team_name = "armo"
SUBMISSION_JSON_FILENAME = str(data_path) + "/solution/solution_armo.json"
# ----------------------------------------

# ana_sub_filename = str(data_path) + "/solution/submission_ana_5_crop_tasks.csv"

ensemble_sub_filename = str(data_path) + "/solution/submission_ensemble.csv"
# ----------------------------------------

import json
import os
import numpy as np
import pandas as pd

def read_sub_csv(sub_filename):
    #df_sub = pd.read_csv(sub_filename, index_col='output_id')
    df_sub = pd.merge(pd.read_csv(SAMPLE_SUBMISSION)[["output_id"]], pd.read_csv(sub_filename), on="output_id", how="left")
    df_sub = df_sub.set_index("output_id")
    df_sub["output"] = df_sub["output"].fillna("")
    return df_sub

sub_ensemble = read_sub_csv(ensemble_sub_filename)

sample_prediction_line = """
|0000035400|0900000067|8000000000|0390006700|0000100001|0100080040|0006000302|0000050010|7000000081|0000040200| |001005|009000|070060|000056|090000|002020| |0080|0000|0002|
"""
guess_determinator = "| |"

"""
        # Generate object solution containing all predictions
        object_solution = {'output_id': id_example, 'number_of_predictions': len(predictions),
                        'predictions': predictions}

        object_prediction = {'prediction_id': prediction_id, 'output': output}

"""

def string2list(pred, kaggle_output_id):
    guesses = pred.strip().split(guess_determinator)
    guesses = [guess.strip() for guess in guesses if len(guess.strip()) >= 3]

    predictions = []
    for prediction_id, guess in enumerate(guesses):
        lines = guess.strip().split("|")
        lines = [line for line in lines if len(line) >= 1]
        output = [[int(e) for e in line] for line in lines]

        object_prediction = {'prediction_id': prediction_id, 'output': output}
        predictions.append(object_prediction)


    output_id = int(kaggle_output_id.split("_")[-1])
    number_of_predictions = len(guesses) # should be 3

    object_solution = {'output_id': output_id, 'number_of_predictions': number_of_predictions,
                           'predictions': predictions}

    return object_solution

solution=string2list(pred=sample_prediction_line, kaggle_output_id="1a6449f1_1")
solution_json = json.dumps(solution)
# print(solution_json)

def get_empty_prediction(output_id):
    predictions = []
    for prediction_id in range(3):
        output = [[0]]
        object_prediction = {'prediction_id': prediction_id, 'output': output}
        predictions.append(object_prediction)

    object_solution = {'output_id': output_id, 'number_of_predictions': len(predictions),
                       'predictions': predictions}
    return object_solution

empty_prediction = get_empty_prediction(output_id=1)

def get_ensemble_prediction(kaggle_output_id):
    pred = sub_ensemble.loc[kaggle_output_id].output
    if (pred is not None) and (len(pred) > 3):
        return string2list(pred=pred, kaggle_output_id=kaggle_output_id)
    return None


# Define function to read tasks
def load_tasks(path):
    """
    Function to load .json files of tasks
    :param path: Path to folder where tasks are stored
    :return: - training and test tasks separated into a list of dictionaries
                    where each entry is of the type {'input': [.task.], 'output': [.task.]}
             - list of file names
    """
    # Load Tasks
    # Path to tasks 
    tasks_path = path
    # Initialize list to s
    # tore file names of tasks
    tasks_file_names = list(np.zeros(len(os.listdir(tasks_path))))
    # Initialize lists of lists of dictionaries to store training and test tasks
    # Format of items will be [{'input': array,'output': array},...,{'input': array,'output': array}]
    tasks_count = len(os.listdir(tasks_path))
    train_tasks = list(np.zeros(tasks_count))
    test_tasks = list(np.zeros(tasks_count))

    # Read in tasks and store them in lists initialized above
    for i, file in enumerate(os.listdir(tasks_path)):
        with open(tasks_path + "/" + file, 'r') as f:
            task = json.load(f)
            tasks_file_names[i] = file
            train_tasks[i] = []
            test_tasks[i] = []

            for t in task['train']:
                train_tasks[i].append(t)
            for t in task['test']:
                test_tasks[i].append(t)

    return train_tasks, test_tasks, tasks_file_names


# Read in evaluation tasks
training_tasks, testing_tasks, file_names = load_tasks(str(test_path))
# Get number of test tasks for outputting progress later and define counter.
num_test_tasks = len(testing_tasks)
counter = 0
# Do some stuff to generate solution
# Allocate space for overall solution
solution = []

kaggle_output_ids = []
# Iterate over all tasks to generate solution
for test_task, task_filename in zip(testing_tasks, file_names):
    # print(f"Predictinng {task_filename} ...")
    # Allocate space for solutions of task examples
    test = []
    # Store filename
    task_name = task_filename.strip('.json')
    # Iterate over test examples (1 or 2)
    for id_example, example in enumerate(test_task):
        kaggle_output_id = task_name.split("/")[-1] + "_" + str(id_example)
        kaggle_output_ids.append(kaggle_output_id)

        object_solution = get_ensemble_prediction(kaggle_output_id)

        if object_solution is None:
            object_solution = get_empty_prediction(output_id=id_example)

        # Add solution of example to list of solutions
        test.append(object_solution)
    # Add solution of examples to overall solution
    object_task = {'task_name': task_name, 'test': test}
    solution.append(object_task)
    # Output progress
    counter += 1
    if counter % 50 == 0:
        print('Generated solution for {} of {} test examples'.format(counter, num_test_tasks))

# Store solution to json file named solution_teamid where our teamid is lab42
# Store it in solution folder which is mounted

solution_json = json.dumps(solution, indent=4)
with open(SUBMISSION_JSON_FILENAME, 'w') as outfile:
    outfile.write(solution_json)


# Print that program has finished
print("-"*40)
print("Solution file is ready!")
