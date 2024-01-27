import os
import shutil
train_data_set = {}
files = os.listdir('./input')
file_count = len(files)
split = file_count//3

os.mkdir('input_1')
os.mkdir('input_2')
os.mkdir('input_3')

folders = ['input_1', 'input_2', 'input_3']
folder_count = 0

file_counter = 0
for file in files:
    if file_counter >= split:
        folder_count += 1
        file_counter = 0
    
    shutil.move(f'./input/{file}', folders[folder_count]+f"/{file}")
    file_counter+=1