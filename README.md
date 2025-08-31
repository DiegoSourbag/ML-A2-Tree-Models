# MLCourse-LU-A2
This repository contains the exercises and questions for assignment 2.

## How to start
* Start by cloning this repository to your computer.
* In your terminal, navigate to the project folder.
* Set up a conda environment with the following commands:

    1. `conda create -n MyEnvName`
    2. `conda activate MyEnvName`
    3. `conda env create -f requirements.yml`
    if this does not work, you can also try using the following command:
    `conda install numpy pandas scikit-learn jupyter pygraphviz`

    Detailed instructions on using conda are available in the Lab 1 notebook (and on the web). 

* Try running `jupyter notebook` from the command line. 
* If it works,you python environment is set up correctly and you can continue on with the assignment. 
* You can either work directly in the jupyter notebook, or you can use an IDE such as VScode. We recommend using VScode. 

### VScode
If you use VScode, make sure you have the jupyter extension installed. You can find the extensions tab in the sidebar of VScode, or access it with the command `ctrl + shift + X`.
Open `A2.ipynb`. In the top right corner you will see the selected kernel (python environment used to run the notebook). Click on this to select a kernel and then click on the option `python environments`. Your conda environment should be listed there. Selecting the environment should allow you to run the notebook. 

If you have difficulty getting the notebook to work, we recommend asking for help during one of the workgroup sessions.
