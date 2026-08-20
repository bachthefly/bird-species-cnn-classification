[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Aju8jnok)
# Lab 5: Advanced Image Classification

## Learning Goals:

This is a team-based assignment designed to function as a follow up to the basic image classification lab and the image clustering mini lab. For this assignment, you may choose your own teammates (3-4 people per team).

Our learning goals include:

1. Using Tensorflow to classify images in Python
2. Using a Convolutional Neural Network (CNN) to classify images
3. Interpreting the results of a CNN classifier  
4. Using multiple pre-trained binary classifiers to make predictions on a set of images

## Background 

This week's lab for DA 351 is based on an article titled "Building a Bird Recognition App and Large Scale Dataset With Citizen Scientists: The Fine Print in Fine-Grained Dataset Collection", which was authored by the SE(3) Computer Vision Group at Cornell University and presented at the Computer Vision and Pattern Recognition (CVPR) Conference in Boston in 2015. They worked with "citizen scientists and domain experts" to develop a "high quality dataset containing 48,562 images of North American birds with 555 categories, part annotations and bounding boxes" (see https://vision.cornell.edu/se3/building-a-bird-recognition-app-and-large-scale-dataset-with-citizen-scientists-the-fine-print-in-fine-grained-dataset-collection/). 

The dataset they created, `NABirds`, is really fantastic, but it's also really huge (9.8 GB compressed on my disk and 9.99 GB after you unzip it)! For our lab, your team can work with this dataset or the Caltech-UCSD Birds-200-2011 Dataset (CUB 200), which was the go-to bird recognition/classification dataset prior to `NABirds`. CUB 200 is still fairly large at 1.2 GB compressed, but that's a lot less than 9.8 GB. 

This means both datasets exceed Github's file size limits, so you will need to download them yourself. Here are the links to the datasets:

| Dataset | Download Link | Filename | 
|---|---|---|
| CUB 200 | https://data.caltech.edu/records/65de6-vp158 | CUB_200_2011.tgz |
| NABirds | https://dl.allaboutbirds.org/nabirds | nabirds.tar.gz |

__Note:__ The `NABirds` dataset requires you to provide your email and agree to the terms and conditions. 


## Assignment Setup

To complete this assignment, you will need to:

1. Choose between the two datasets, download the data, and unzip the compressed files
2. Verify your installation of Tensorflow version 2.13 (other versions may work)
3. Set up files and folders for classification (by hand or computationally)
4. Read the data into Python as a training set and a validation set using Tensorflow's `image_dataset_from_directory` method

## Choosing a Dataset 

As stated, the size of `NABirds` is the most important factor to consider, but they are also organized in slightly different ways. Both datasets are organized with one particular bird species per folder, but the CUB 200 dataset has a unique id number and a species name in each folder name, whereas the `NABirds` dataset uses a numbering system for folders, and you have to look up the species name in the file `classes.txt`. 

## Set up Files and Folders

Tensorflow's `image_dataset_from_directory` method is _much_ faster than openCV, but it requires your files and folders to be set up in a specific way. You will need move files around as you go to create the following structure:

```
> parent_folder
	> class_a_folder
		class_a_img_1
		class_a_img_2
		etc.
	> class_b_folder
		class_b_img_1
		class_b_img_2 
		etc.
```

Once you have this structure, you can use it to define training and validation sets. Labels can be supplied, but it's much easier to have Tensorflow infer them from the folder names. 

In this lab, you will spend most of your time doing setup, because your main task is to train and evaluate multiple binary classifiers. You will identify ten bird pairings and write a hypothesis about which pairs will be the hardest to differentiate from one another. You should select a variety of pairings based on how difficult you think they will be to tell apart. For example, if you use the CUB 200 dataset, you may choose pairs such as:

1. California Gull vs. Artic Tern (medium?)
2. Black-footed Albatross vs. Sooty Albatross (hard?)
3. Arctic Tern vs. Boat-tailed Grackle (easy?)

Please do not repeat bird species anywhere in this setup. Each pair will become the basis for a binary classification task below, and you'll use all ten models to describe the characteristics of a novel bird class. These are hypotheses, so it's okay if performance differs from your expectations. 

__Note:__ Due tot he size of the source data and the required Python libraries, this assignment may prove difficult to complete on some students' personal computers. In class, we will all sign up for accounts with the Ohio Supercomputer Center (OSC), which offers cloud-based resources for students at Denison. You may choose to run this lab through the OSC's Jupyter app, but you will still need to turn in your assignment using Github and Canvas. More details about the OSC will be covered in class.

### Train and Evaluate Your 10 Models

Once your setup is complete, you will train and validate binary classification models for all of your pairs. Every model should be a CNN with the same architecture. (Your team can choose that architecture, but we will discuss some common options in class.) For each model, you will report the following results:

1. Overall accuracy of the model
2. A confusion matrix of your validation set's True Positives, True Negatives, False Negatives, and False Positives 
3. Per class precision and recall (using `scikit-learn` functions)

__Note:__ You should use a different variable name for every model so that all ten models are in memory once you have trained them all. 

### Use Binary Models to Describe Novel Data

Choose a bird species that doesn't appear anywhere in your ten pairings from above. Read images for this species into a Tensorflow `Dataset` and use each of your ten models to predict the novel species' probability of class A or B. Definitionally, this new species will belong to neither class but, at the end of this process, you'll have a bunch of data on the new bird species that looks something like this:  

| Image | Gull Probability | Black-footed Albatross Probability | Arctic Tern Probability | 
|---|---|---|---| 
| Image 1 | 0.12 | 0.55 | 0.72 | 

Generate a pandas DataFrame like the example above for all photos of your chosen bird species and all ten of your classifiers. Display the results of your __novel bird species DataFrame__ (`novel_species_df`) in your lab report. 

### Repeat and Classify

Choose a second novel species and read images for this species into a Tensorflow `Dataset`. Use each of your ten models to predict the novel species' probability of class A or B and add the results to the __novel bird species DataFrame__. Keep track of which rows belong to novel species 1 and which rows belong to novel species 2. Using scikit-learn, set up a traditional binary classifier (KNN, logit model, etc.) using `novel_species_df` as your predictors (`X`), and using each bird's true species as the class labels (`y`). Perform a train/test split on the data, train and test your model, and report the following results:

1. Overall accuracy of the model
2. A confusion matrix of your validation set's True Positives, True Negatives, False Negatives, and False Positives 
3. Per class precision and recall (using `scikit-learn` functions)

## Lab Report Components 

__Title__	

Brief and informative, gives some idea of your core question or topic area. This block should also include team name, all student names, and the date of the submission.

__Introduction__

- One or two paragraphs providing an overview of the data set and the purpose of the report. Where did the data come from and what does it show? What questions are you trying to answer?

	- Make sure you state your ten pairs of bird species, with a hypothesis for each pair about how a binary classifier will perform. Explain why you chose the pairs that you chose. 

- One paragraph discussing the stakeholders in your data analysis and your ethical concerns or responsibilities using the data and in your analysis. Everyone has ethical considerations, no matter what the dataset or subject matter!

__Code & Results__

- This section should include numerous chunks of python code with helpful annotations (comments and/or brief markdown cells after each chunk). Remember to print or return all results in your notebook so they display in the html file you output for the assignment. 

- For this assignment in particular, this section is where you will train and validate binary classification models for all ten of your bird species pairs, use binary models to describe two novel bird species, and train and test a __novel bird classifier__ using your ten models as an ensemble.  

__Interpretation__ 

Interpret your output, highlighting the key results and explaining the main takeaways. For this lab, make sure you address the following:

1. Revisit your hypotheses from the introduction. How did your models do compared to your hypotheses? What was surprising and why?
2. Discuss the precision and recall scores of each class for your various models. What seems to be your strongest model and why?
3. Discuss the results of your __novel bird species DataFrame__. Do the classifier scores for the novel species seem to describe them in a patterned way? Why or why not? 
4. Fully interpret the results of your __novel bird classifier__. Did it perform better worse worse than you expected? Was there anything surprising about your results? 

__Conclusion__

- One or two paragraphs summing up what you've learned. What does this analysis tell us? What are the strengths/limitations of  this data set? What are the strengths/limitations of this method? What is one future direction you could envision for future data analysts or data collectors? 

- Additionally, take a step back and analyze your own use of code. Provide some rationale for choices you've made. Considerations may include performance, human readability, code dependencies, and reproducibility.





