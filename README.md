## COMP 598 - Applied Machine Learning - Project 3

The	goal of this project is to devise a	machine learning algorithm to automatically classify images of hand written digits (from 0 to 9) represented in cropped image.

These are the algorithm we used:
 - (1) A baseline learner consisting of logistic regression.
 - (2) A linear SVM.
 - (3) A fully connected feedforward neural network trained by backpropagation, where the network architecture (number of nodes / layers), learning rate and termination are determined by cross‐validation.


# Requirements:

Python requirements:

    csv
    pickle
    numpy
    sklearn
    scipy
    matplotlib
    ...


# To run the algorithms:

 - Logistic Regression:

    + File structure requirements:
    ```
        logisticRegression.ipynb
        test_inputs.csv
        train_inputs.csv
        train_outpus.csv
    ```
    
    + Use Jupyter Notebook to view the following file: `logisticRegression.ipynb`


 - SVM:
    
    + File structure requirements:
    ```
        nicoLoadTestData.py
        nicoSVM.py
        data_and_scripts/
            test_inputs.csv
            train_inputs.csv
            train_outpus.csv
        obj/
            test.pkl
            train.pkl
    ```

    + Check nicoSVM for parameter selection, and when ready run the following command: ```python nicoSVM.py```


 - Neural Network:



# Authors:
 - Andres Felipe Rincón
 - Ryan Razani
 - Nicolas Angelard-Gontier

