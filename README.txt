Dataset download :
To download the dataset do the following : 
1.	Go to https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2.	Download the dataset
3.	Place the entire dataset in a folder called input
4.	Copy all images from HAM10000_images_part_2 into HAM10000_images_part_1
5.	Run the file submission_code/dataset_splitting/splitting/dataset_split.py
6.	This should separate the test set into a folder called test_images

To run python files : 
Note : Files were being run in the STONE virtual machine 
1.	Create a python virtual environment
2.	Activate the environment
3.	Install the requirements listed in submission_code/requirements.txt
4.	Run files using python3 file_name.py

To load the webpage : 
1.	Activate the virtual environment (with the installed requirements)
2.	Navigate to submission_code/webpage/flask
3.	Run the flask file using flask run
4.	Open Firefox and go to URL : http://localhost:5000/

File Structure :
submission_code
-	README.txt
A file containing instructions on how to run the code, file structure and references
-	requirements.txt
Requirements to run the code, must be installed for it to run
-	dataset_splitting
-	splitting 
-	dataset_split.py
Code splits dataset into test and training set. The test set is moved into test_images/test_data 
-	distillation_model
-	distill 
-	fold_x
Results of the teacher model
-	fold_distilled_x
Results of distilled model
-	distillation_teacher_crossval_cleaned.py
The distillation code which trains the teacher and student model
Everything is done without modules for clarity on distillation process
Also can do hyperparamter, uncomment commented code and switch with current eval
Reference for understanding how to load ham10000 data, its structure and model training: 
https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training
-	hyperparam_optimisation
-	evaluation_metrics.txt
Results from hyperparam optimisation
-	optimum_threshold.py
Outputs evaluation results by threshold to command line

-	modules
-	data_processing_clean.py
Preprocessing of an image directory to convert it to a data frame, training or validation set. 
Reference for understanding how to load ham10000 data and its structure: https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training 
-	evaluation_utils.py
Calculating evaluation metrics of a model.
-	model.py 
Initializing a model 
Reference for understanding how to modularise model loading: https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training 
Reference for understanding CNN construction in Python
https://gitlab.com/muhammad.dawood/lupi_pytorch/-/tree/master/Distillation%20LUPI%20paper%20experiments?ref_type=heads 
-	training_clean.py 
Training a model 
Reference for understanding model training in python : 
https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training

-	preproccesing 
-	baseline
No preprocessing results
-	complex_preproccesing_results
Complex preprocessing results  
-	simple_preprocessing_results
Simple preprocessing results 
-	complex_preprocessing.py 
Complex preprocessing code  
-	preproccesing_evaluation_baseline.py
Evaluation of model performance on raw data
-	preproccesing_evaluation.py
Evaluation of model performance of complex and simple preprocessing 

-	student_model
Note : Data loading is done separately as preprocessing, especially resizing, varies for models.
-	student_1
-	 fold_x
contains results for fold x
-	student_1.py 
Train student 2 model
-	student_2
-	 fold_x
contains results for fold x
-	student_2.py 
Train student 2 model
-	student_3
-	 fold_x
contains results for fold x
-	student_3.py 
Train student 3 model

-	teacher_model
-	ensemble_learning 
-	model_name 
Folders containing results for models and voting based models
-	ensemble_learning_clean.py
Code for ensemble learning experiment for teacher models 
Reference for understanding voting : 
https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training


-	test_set_eval
NOTE : every directory contains a copy of the models output from cross vlidation
-	distillation
-	fold_x
Contains results of model from folds x results on test set
-	test_eval.py
Code for running evaluation on test set
-	teacher
-	fold_x
Contains results of model from folds x results on test set
-	test_eval.py
Code for running evaluation on test set
-	student
-	fold_x
Contains results of model from folds x results on test set
-	test_eval.py
Code for running evaluation on test set