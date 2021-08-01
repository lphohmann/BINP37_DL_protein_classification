# Readme file: Bioinformatics research project (BINP37)
## Topic: Evaluating the potential of a deep learning-based approach for the functional annotation of microbial proteins.

**Software basis:**
- Python programming language (*version 3.8.3*)
- NumPy (*version 1.19.5*)
- scikit-learn (*version 0.24.2*)
- PyTorch (*version 1.9.0*)
- Fastai (*version 2.4.1*)

## **Introduction:**

In elucidating microbial climate change-driven adaptations, the functional annotation of proteins plays a critical role. In recent years the application of deep learning (DL), a specific class of supervised machine learning algorithms, has seen a large increase in their usage for many bioinformatic analysis pipelines. Therefore, this project aimed to evaluate the potential of developing a DL-based tool for the functional annotation of microbial proteins sequences as an alternative to established sequence similarity-based approaches. To that end a deep learning classifier with a ResNet-152 architecture was constructed and trained using labelled protein sequences obtained from the KEGG database.

## **Files included in this repository**:
**1. data_preprocessing.ipynb:**
Google colab notebook file containing the code for preprocessing the original dataset. Using it, a test set as well as a second dataset (called trainval) are created. It is to be run in an interactive session on the Google colab platform.

**2. 1D_ResNet.ipynb:**
Google colab notebook file containing all code to load the training data, define the model architecture, train the model, and evaluate its performance on the training set. It is to be run in an interactive session on the Google colab platform.

**3. tetralith_script_1d_resnet152.py:**
Python script for running a job on the Tetralith HPC cluster that contains the code to load the training data, define the model architecture, train the model, and evaluate its performance on the training set. It differs from the 1D_ResNet.ipynb file in that several lines were excluded that served the purpose of debugging and checking steps in an interactive session.

**4. myjob.sh:**
Bash script specifying the details of the job to be run on the NPC cluster. It has to be specified when submitting the batch job.

## **Workflow:**

**1. Setting up the notebooks in Google Colaboratory**

1. Log into your google drive.
2. Click on Google Colaboratory by clicking on the "New" button in the upper left corner and then hovering your mouse pointer over "more" for it to appear.
3. An empty notebook will open and you'll have to click on "File" and then "Upload notebook".
4. In the pop-up window select the GitHub tab and paste the URL https://github.com/lphohmann/BINP37_Research_project into the search bar.
5. Below the bar, all notebooks in the repository should appear and you can click on the one you want to import.
6. Before running a notebook you can specify the runtime type you want to use by clicking on ‘Runtime’ and selecting ‘Change runtime type’. If you want to run the notebook using a GPU (required for model training in the notebook *1D_ResNet.ipynb*), choose GPU in the drop-down menu of the pop-up window.

**2. Required directory structure in the working directory**

In the working directory on google drive (from where you run the notebooks), a directory named "data" with the original dataset should be present.

**3. Data preprocessing overview**

***File:*** *data_preprocessing.ipynb*

1. Set-up consisting of mounting the google drive so that the data directory can be accessed and importing required modules.
2. Creating a pandas dataframe from a fasta file with the protein sequences and associated functional annotations (in form of K numbers).
3. Performing a random stratified split by K number of the dataframe thereby producing test and trainval set which are saved as .csv files in the data directory.

*Note: Detailed step descriptions are found in the notebook file*

**4. Interactive model development overview**

***File:*** *1D_ResNet.ipynb*

1. Set-up consisting of installing the Fastai library, mounting the google drive and importing required modules.
2. Create the DataBlock which serves as a template on how to load data from a dataset and transform it into the correct format for serving as model input.
3. Create the DataLoaders which collates individual fetched data samples into batches. Additionally, the training batches are balanced by oversampling classes with fewer associated sequences.
4. Create the Learner which handles the training loop of the model. The step included specifying the metric for model evaluation, the loss function, and adding callbacks that save models during training and stop the training in case the model doesn't keep improving.
5. Defining the model architecture. The final architecture of the model in this project is commonly known as a ResNet-152.
6. Finding a suitable learning rate and subsequent model training using one-cycle training.
7. Model evaluation by loading the test set and subsequently calculating the metric on the test set.
8. Visualization of the models's classification performance by creating a confusion matrix based on the predictions in the test set.

The files that will be created by running the notebook are:
- ***export.pkl*** : File with the final model and the learner (template on how to load data) that is used for loading the model for inference steps.
- ***test_confusion_matrix.pdf*** : PDF file with the test set-based confusion matrix.
- ***1D_ResNet152.pth*** : File with the final model. The notebook automatically creates the directory "*models*" that contains the file in the working directory.

*Note 1: Detailed step descriptions are found in the notebook file. I recommend running the notebook as the interactive sessions on google colab facilitate gaining insight into how the model is built and trained.*

*Note 2: This notebook is not essential to be run if the following step on the Tetralith HPC cluster is executed but it serves as a great way to understand the workflow due to the interactive nature of colab notebooks making is possible to explore what is happening at each step. It is important to clarify that running this notebook does not replace the next step as it is not possible to train for many epochs on google colab.*

**5. Training on the Tetralith HPC cluster of the Swedish National Supercomputer Centre (NCS)**

***Files:*** *tetralith_script_1d_resnet152.py* ; *myjob.sh*

1. Getting access to the cluster by following the detailed information provided by the NCS at https://www.nsc.liu.se/systems/tetralith/.
2. Copying required files for the model training to your home directory on the server.

```bash
# uploading the data directory containing the previously produced test.csv and trainval.csv to the home directory on the server
scp -r data/ x_lenho@tetralith.nsc.liu.se:
# uploading the python script with the essential code for creating and training the model
scp tetralith_files/tetralith_script_1d_resnet152.py x_lenho@tetralith.nsc.liu.se:
# uploading the bash script specifying the details of the job to be run on the NPC cluster
scp tetralith_files/myjob.sh x_lenho@tetralith.nsc.liu.se:
```
3. Logging onto the server and using conda to create an environment with the required software (no need to install conda as it is already available).

```bash
# login
ssh x_lenho@tetralith.nsc.liu.se
# load the anaconda module with the newest python version
module load Python/3.8.3-anaconda-2020.07-extras-nsc1
# create a new environment and activate it
conda create -n DLproject python=3.8.3
source activate DLproject
# install fastai; all required software is automatically installed with fastai
conda install -c fastchan fastai=2.4.1
```

*Note: After logging out, the next time you log on you activate your environment as follows:*
```Bash
# logging in after the first time and loading the environment
$ module load Python/3.8.3-anaconda-2020.07-extras-nsc1
$ source activate DLproject
```

4. It is recommended to test the script in an interactive session to make sure it runs without errors before submitting it as a batch job. To do so it doesn't make sense to run the model for a lot of epochs as this will be done as a batch job. Therefore, I recommend changing the specified training epochs in the script to epochs=1 and after making sure it runs fine, change it back to 200.

```bash
# first make the uploaded scripts executable
chmod +x tetralith_script_1d_resnet152.py
chmod +x myjob.sh
# change the training epochs to 1
nano tetralith_script_1d_resnet152.py
# run an interactive job
interactive -n 1 -c 32 --gpus-per-task=1 -t 00:30:00 # allocates 1 task comprising 32 CPU cores and 1 GPU for 30 minutes.
./tetralith_script_1d_resnet152.py # run the script
# if everything works as expected change back to epochs=200
nano tetralith_script_1d_resnet152.py
```

5. Now, a batch job is submitted that trains the model for 200 epochs. Running a batch job on Tetralith produces an additional output file named *slurm-NNNNN.out* in the directory where you submitted the job. It contains the the standard output and standard error from the job script. Make sure that you activated your conda environment where the required software is installed in before submitting the batch job as it will be recorded and used when the job starts.

```bash
# submit the batch job
sbatch myjob.sh
```

Further information on running jobs on Tetralith and using GPU nodes can be found at:

https://www.nsc.liu.se/support/batch-jobs/introduction/

https://www.nsc.liu.se/support/running-applications/

https://www.nsc.liu.se/support/systems/tetralith-GPU-user-guide/

## **Additional resources:**

For anybody that wishes to get familiar with building deep learning models, I highly recommend the course "*Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD*" by Sylvain Gugger and Jeremy Howard. You can find it at: https://course.fast.ai/.
