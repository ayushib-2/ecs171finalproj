# ECS 171 Final Project - Stroke Prediction
**Group 21: Ayushi Bhatnagar, Kaitlyn Li, Brandon Glawitsch, Alex Ngo**

Strokes, which are one of the critical health issues and a leading cause of death worldwide, can lead to significant disability or death. 
It is an acute neurological disorder in which blood in the brain suddenly stops, leading to brain cells dying as they are deprived of
oxygen and could lead to disability depending on which area of the brain was affected.
The World Health Organization (WHO) reports that strokes account for 11% of global deaths
and as the second leading cause of death. With this in mind, our project aims to leverage
clinical features such as smoking status, gender, age, and other factors to predict the likelihood of
someone experiencing a stroke. We feel that early detection can reduce the impact
of strokes, and by employing machine learning techniques, we believe that we can contribute to
the knowledge that supports stroke prevention and help spread awareness about this acute disorder.

## Dataset
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Dependencies
* Languages: Python
* Libraries: NumPy, pandas, Matplotlib, seaborn, collections, sklearn, Plotly

## Usage
### Models
* Logistic Regression
* SVM
* Random Forest
* Neural Network

### How to run our model
Run `python web.py` to launch the Flask server for the frontend on port 5000. Then, go to localhost:5000 to view it.