import file_io as fio
import answers as ans
import librosa
import os
import numpy as np
import random
import dataset_class as dc

# This script performs the execution of the functions provided in Exercise 1 and
# the ones implemented by the student. Please ensure the correct directories are used.
# The TA will not debug your code.

#### TASK 2.1
#### Place the execution of task 2.1 below
os_files = fio.get_files_from_dir_with_os("/Users/vudinhthi2304/Desktop/COMPSGN220/Exercise1/audio")
pathlib_files = fio.get_files_from_dir_with_pathlib("/Users/vudinhthi2304/Desktop/COMPSGN220/Exercise1/audio")

print("Files retrieved using os:")
print(os_files)

print("Files retrieved using pathlib:")
print(pathlib_files)

sorted_os_files = sorted(os_files)
sorted_pathlib_files = sorted(pathlib_files)

print("Sorted files using os:")
print(sorted_os_files)

print("Sorted files using pathlib:")
print(sorted_pathlib_files)

# a) Difference in the paths returned by each function:
# os: The paths are returned as file names without any additional information 
# about the directory structure. For example, '10.wav'.
# pathlib: The paths are returned contain the full path information, including 
# the directory structure. For example, PosixPath('/Users/vudinhthi2304/Desktop/COMPSGN220/Exercise1/audio/10.wav').

# b) Difference in the values returned by each function:
# os: The values are just the names of the files, returned as plain strings.
# pathlib: The values are more structured, returned as PosixPath objects that 
# represent the full file path, including the directory path.

# c) Using the sorted() function in Python, sort the paths. What do you observe in the ordering of the paths?
# When sorting file names with the sorted() function, it performs a lexicographical (dictionary) sort. 
# In lexicographical order, string sorting happens based on character comparison, meaning with letters from left to right.
# Therefore, for example, '10.wav' comes before '2.wav' because '1' comes before '2' in the ASCII character set.

# d) Without changing the conceptual name of the files (e.g. a file named 0001.wav can
# be changed to 001.wav but not to 0002.wav), how can you fix what you observed from the ordering of the files?
# To fix the ordering, you can use zero-padded numbering. This ensures that the numbers in the filenames are sorted 
# in a more intuitive way, such as treating the numbers as integers instead of strings. We can use the zfill() method 
# to pad the numbers with leading zeros so that all filenames have the same length, making them sort correctly as numbers.
# For example, '1.wav' can be renamed to '001.wav', '2.wav' to '002.wav', and so on. This will ensure that the files are
# sorted correctly when using the sorted() function.


#### TASK 2.2
#### Place the execution of task 2.2 below
for file_path in pathlib_files:
    audio = ans.get_audio_file_data(file_path)


# Compute the duration of each audio file in seconds

#### TASK 3.1
#### Place the execution of task 3.1 below
ex1 = "/Users/vudinhthi2304/Desktop/COMPSGN220/Exercise1/exercise_01/ex1.wav"
ex1_mel = ans.extract_mel_band_energies(ex1)

# a) The first dimension represents the mel frequency bands. These bands correspond to 
# different frequency bins in the Mel scale. In this case, there are 40 mel frequency bands.
# b) The second dimension represents the time steps or frames. Each frame corresponds to a short segment of the audio signal.
# c) There is a connection between the length of the corresponding audio file and the extracted features, 
# specifically the time dimension of the feature array. The number of time frames (582 in your case) is 
# determined by the length of the audio file, the frame size, and the hop size. A longer audio file will generally result in 
# more time frames because the audio is split into smaller overlapping segments. The frame size means how much of the audio 
# is analyzed in each window, and the hop size is how much the window shifts for each new segment. These parameters define 
# how the audio is sliced into segments. Otherwise, the mel frequency bands are fixed and do not depend on the audio file length.
# d) Quiet sections of the audio will show darker areas with little energy. Loud or dynamic sections shows brighter regions
# in specific bands, depending on the dominant frequencies at that time. Repeated patterns in the figure may correspond to
# rhythmic elements in the audio .


#### TASK 3.2
#### Place the execution of task 3.2 below
output_dir = "/Users/vudinhthi2304/Desktop/COMPSGN220/Exercise1/serialized_dictionaries"
os.makedirs(output_dir, exist_ok=True)

for file_path in pathlib_files:
    mel_spec = {
        'file_path': f"{output_dir}/{file_path.stem}.pkl",
        'features': ans.extract_mel_band_energies(file_path),
        'class': random.randint(0, 1)
    }
    ans.serialize_features_and_classes(mel_spec)

#### TASK 4.1
#### Place the execution of task 4.1 below
dataset = dc.MyDataset(output_dir)


#### TASK 4.2
#### Place the execution of task 4.2 below
ans.dataset_iteration(dataset)
# batch_size controls the number of samples per batch.
# shuffle=True shuffles the data before creating batches, to reduce bias introduced by the order of data
# drop_last=True drops the last incomplete batch if the dataset size is not divisible by the batch size.