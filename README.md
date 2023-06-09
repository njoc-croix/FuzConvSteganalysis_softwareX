FuzConvSteganalysis: Steganalysis via Fuzzy Logic and Convolutional Neural Network

FuzConvSteganalysis is an innovative software that uses a combination of fuzzy logic and convolutional neural networks to detect hidden information in digital images. It focuses on locative steganalysis, identifying the exact location of concealed data in images. FuzConvSteganalysis employs advanced machine learning techniques, providing a robust tool to uncover covert information hidden in digital images. This software has the potential to prevent the misuse of digital images for data theft and covert communication, enhancing digital security.

Citing

If this work is utilized for your research or if you discover value in our long paper with all results, kindly acknowledge the work by including a citation of our long paper at: 

https://www.sciencedirect.com/science/article/pii/S1110866523000336?via%3Dihub and cite as: 

N. J. De La Croix and T. Ahmad, “Toward secret data location via fuzzy logic and convolutional neural network,” Egyptian Informatics Journal, vol. 24, no. 3, p. 100385, Sep. 2023, doi: 10.1016/j.eij.2023.05.010.

Folders in this project: 

	Matlab_codes: 
	
	This folder contains two files with one entitled: “My_WOW_main_function.m”. The file includes the codes for data embedding with WOW (steganographic algorithm). The second file is entitled: “WOWupdate.m” which is a main function to run the data embedding.
	
	sampledatasetimagepixels: 
	
The sample dataset folder contains 33 image pixels ttwo main subfolders for sample innocent pixels padded, and sample altered pixels padded to work with the model. It is worth noting that the images used in experiment have been got from BOSSBase 1. 01 (http://agents.fel.cvut.cz/boss/index.php?mode=VIEW&tmpl=materials).
	
	Numpy_files:
	
This folder contains six files in .numpy extension which encompass the sample datasets with their respective labels for training, testing, and validation. 
	
	SRM_Filters:
	
This folder contains a file in .numpy which contains spatial rich models used in this work. 
	
	Main_file:
	
This folder contains a Jupyter notebook entitled “FuzConvTrue” for running the main code of our work.
	
	Best_trained_model:
	
This folder contains the best model of our trained models with WOW algorithm under the payload size of 0.1bpp.


Requirements to run the code:

This work was done using Python 3 and the following libraries and frameworks are necessary to run it: TensorFlow, os, scikit-image, NumPy, OpenCV, Matplotlib, Time, and glob.

Notice: 

Before running the notebook in the main_file, please verify that the file paths are correct, and all libraries are installed.


Thank you for considering our code and giving suggestions on how to make it more user-friendly!





