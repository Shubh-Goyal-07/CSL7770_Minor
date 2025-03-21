# CSL7770 Minor Examination - Shubh Goyal (B21CS073)

The repository contains the code and report for the Minor Examination of the course CSL7770 - Speech Understanding.

**Directory Structure**
```
.
├── Q1
│   ├── dataset
│   └── main.py
├── Q2
│   ├── Speeches_of_leaders
│   ├── Comparative_speeches
│   └── main.py
├── Q3
│   ├── dataset
│   ├── main.py
│   └── model.py
├── requirements.txt
├── report.pdf
└── README.md
```

The report for the assignment is present in the `report.pdf` file.

The code for each question is present in the respective directories.

The `requirements.txt` file contains the list of required packages to run the code.


**How to use this repository**

First, make sure in the root directory of the repository.

Now, prepare the environment by installing the required packages using the `requirements.txt` file.

```bash
conda create -n csl7770 -y
conda activate csl7770
conda install --file requirements.txt -y -c conda-forge -c nvidia -c pytorch
```

The above command will create a new conda environment named `csl7770` and install all the required packages in it. It will take a few minutes to complete.


## Question 1

Navigate to the `Q1` directory to run the code for Question 1.

```bash
cd Q1
```

The directory structure is as follows:
```
.
├── dataset
├── logs.log
└── main.py
```

The `dataset` directory contains the dataset for the question.

The `logs.log` file contains the logs generated during the execution of the code.

The `main.py` file contains the code for the question.

To run the code, execute the following command:

```bash
python main.py
```

The code will create two directories named `characteristics` and `plots` in the current directory.

The `characteristics` directory will contain the json files for the characteristics of the dataset. The file names will be same as the names of the files in the dataset.
There will be a `all_characteristics.json` file which will contain the characteristics of all the files combined.

The `plots` directory will contain the plots for the characteristics of the dataset. There are directories for each file in the dataset. The plots for the characteristics of each file are saved in the respective directories.
The following plots are generated:
1. Waveform
2. Amplitude Envelope
3. Pitch Envelope
4. RMS Envelope
5. Spectrogram with Fundamental Frequency (F0) Contour


## Question 2

Navigate to the `Q2` directory to run the code for Question 2.

```bash
cd Q2
```

The directory structure is as follows:
```
.
├── Comparative_speeches
├── Speeches_of_leaders
├── logs
└── main.py
```

The `Comparative_speeches` directory contains the two speeches for comparison.

The `Speeches_of_leaders` directory contains the speeches of the leaders (the dataset).

The `logs` directory contains the logs generated during the execution of the code.

The `main.py` file contains the code for the question.

To run the code, execute the following command:

```bash
python main.py --dir <path_to_directory>
```

Replace `<path_to_directory>` with the path to the directory containing the dataset.
If no path is provided, the code will use the default dataset present in the `Speeches_of_leaders` directory.

The code will create a directory named `features` in the current directory. If custom path is provided using the `--dir` argument, the directory created will be named `features_<directory_name>`.

The directory will contain the json files for the features of the dataset. The file names will be same as the names of the files in the dataset.
There will be a `ALL_FILES_FEATURES.json` file which will contain the features of all the files combined.


## Question 3

Navigate to the `Q3` directory to run the code for Question 3.

```bash
cd Q3
```

The directory structure is as follows:
```
.
├── dataset
├── main.py
└── model.py
```

The `dataset` directory contains the dataset for the question (downloaded from the given link).

The `main.py` file contains the code for feature extraction.

The `model.py` file contains the code for training and testing the classifier.

To run the code, execute the following command:

```bash
python main.py
```

The code will create a file named `features.csv` in the current directory, containing the features extracted (f0, f1, f2, f3) along with the file name and vowel label.

Also, it will create a file named `vowel_space.png` in the current directory, containing the F1-F2 vowel space plot (Color-coded by vowel label).


Now, run the following command to train and test the classifier:

```bash
python model.py --classifier <classifier_name>
```

Replace `<classifier_name>` with the name of the classifier to be used. The available classifiers are `svm`, `knn`, `dt` and `gmm`.
If no classifier is provided or `all` is provided, the code will train and test all the classifiers.

The code will create a directory named `models` in the current directory. The directory will contain the trained models for each classifier.

The code will also create a directory named `confusion_matrix` in the current directory. The directory will contain the confusion matrix for each classifier.

Finally, the code will generate a file named `model_accuracies.png` in the current directory, containing the train and test accuracy plots for each classifier.