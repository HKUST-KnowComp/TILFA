"""
=========================================================
Title: ImageArg Shared Task Code - Check Submission Format
---------------------------------------------------------
Please be careful about your submission file name and file 
format, which will be used to validate your submission.

File Format Validation:
The file should be in CSV format with a maximum of two 
columns: "tweet_id", and either "stance" or "persuasiveness". 

TaskA CSV file should include:
tweet_id, stance
1204712613394944000, oppose
1204712613394933000, support

TaskB CSV file should include:
tweet_id, persuasiveness
1204712613394944000, yes
1204712613394933000, no

Please note tweet id should be integer, and
stance/persuasiveness should be lower cases.

---------------------------------------------------------
File Name Validation:
1. Team_Name: your team name
2. Method_Name: give a name to your model/method 
   (we accept multiple methods/models you might try)
3. Task_Name: shared task name (For Subtask A Stance 
   Classification, the task name is "TaskA", while for 
   Subtask B Persuasiveness Classification the task 
   name is "TaskB")
4. Attempt_Number: the number of submission attempts
   (you can do a total of 5 submission attempts)
5. No period symbol in Team_Name, Method_Name, Task_Name,
   and Attempt_Number.

Your file name should be like 
"{team_name}.{method_name}.{task_name}.{attempt_number}.csv"
i.e., test-team.baseline.TaskA.1.csv.
---------------------------------------------------------
Notice: This code is managed by ImageArg Shared Task
(https://imagearg.github.io/).
---------------------------------------------------------
Data: 2023-07-16
=========================================================
"""

import pandas as pd
import argparse
import sys

sys.tracebacklimit = 0


def verify_filename(file_name):
    try:
        team, model, subtask, run_number, file_format = file_name.split('.')
    except:
        print("Wrong File Name Format!")
        return

    if not subtask in ["TaskA", "TaskB"]:
        print("Wrong Task Name Format!")
        return

    if file_format.lower() != "csv":
        print("Wrong File Name Format!")

    try:
        if int(run_number) >= 6:
            print("Wrong Attempt Number!")
            return
    except:
        print("Wrong Attempt Number!")

    print("File name is good to go!")

    return True


def verify_format(file_name):
    team, model, subtask, run_number, file_format = file_name.split('.')
    predictions = pd.read_csv(file_name)

    if len(predictions.columns) != 2:
        print("Only include two columns: tweet_id and stance/persuasiveness columns!")
        return

    if len(predictions) != 300:
        print("Missing predictions! Should be 300 examples!")
        return

    if "tweet_id" not in predictions.columns:
        print("Wrong column name in your file!")
        return

    if subtask == 'TaskA':
        try:
            if "stance" not in predictions.columns:
                print("Wrong column name in your file - should be named stance")
                return
            assert all(_pred in ("support", "oppose") for _pred in predictions['stance'].tolist())
        except:
            print('Wrong values for Subtask A Stance, please use "support", "oppose" labels in your predictions')
            return
    elif subtask == 'TaskB':
        try:
            if "persuasiveness" not in predictions.columns:
                print("Wrong column name in your file - should be named persuasiveness")
                return
            assert all(_pred in ("yes", "no") for _pred in predictions['persuasiveness'].tolist())
        except:
            print('Wrong values for Subtask B Persuasiveness, please use "yes", "no" labels in your predictions')
            return
    else:
        print("Wrong Task Name!")
        return
    print('File Format is good to go!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='imagearg shared task', description='check submission file format')
    parser.add_argument('filename')
    args = parser.parse_args()

    res = verify_filename(args.filename)
    if res:
        verify_format(args.filename)
