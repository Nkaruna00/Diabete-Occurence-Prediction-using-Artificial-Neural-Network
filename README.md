#  Diabete Occurence Prediction using Artificial Neural Network 

Neural Network made with Keras and Python to predict the diabete occurence in a population.



## Description


Development of an artificial neural network to predict whether a patient is diabetic or not, based on certain diagnostic measures included in the data set.  In particular, all patients here are females at least 21 years old of Pima Indian heritage.  

This dataset comes from the National Institute of Diabetes and Digestive and Kidney Diseases.

The dataset consists of several medical predictor variables.The predictor variables include:

* Number of pregnancies the patient has had
* Glucose level
* Blood pressure
* Skin thickness
* BMI
* Insulin level
* Diabete Pedigree function (scores likelihood of diabetes based on family history)
* Age

and a target variable:
* Outcome (1 if the patient developed diabetes, 0 otherwise)  


The prediction variables are first scaled to avoid scaling differences between values and to facilitate learning.
The dataset is then separated with 80% of the dataset used for training and 20% of the dataset used to test the model.  


* The first layer is composed of 64 neurons with a Relu activation function and a Dropout of 0.2 to avoid overlearning  


* The second layer is composed of 32 neurons with a Relu activation function and a Dropout of 0.2 to avoid overlearning  

* The third layer contains 1 neuron with a sigmoid activation function to estimate between 0 and 1 the prediction of the occurrence of diabetes in a patient. 


The optimizer used in this model is Adam.
The model is saved with its weights in the file diabetes.h5 file.  
The prediction of the model is 82%.

## Getting Started

### Dependencies

* PIMA Diabete Dataset
* Python3
* Keras with Tensorflow backend


### Executing program

* Run diabetes.py
```
python3 diabetes.py
```


## Author

KARUNAKARAN Nithushan

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

