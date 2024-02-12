
# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
# Library to split data
from sklearn.model_selection import train_test_split
# Library to encode the variables
from sklearn import preprocessing
# To plot confusion matrix
from sklearn.metrics import confusion_matrix
# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# library to import to standardize the data
from sklearn.preprocessing import StandardScaler
#To import different metrics
from sklearn import metrics
from tensorflow.keras import backend
# Library to avoid the warnings
import warnings
warnings.filterwarnings("ignore")
# importing different functions to build models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
# importing GridSearch CV
from sklearn.model_selection import GridSearchCV
# importing roc_curve to plot
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# importing SMOTE
from imblearn.over_sampling import SMOTE
# importing metrics
from sklearn import metrics
import random
#Importing classback API
from keras import callbacks

# step 1: import dataset 
ds = pd.read_csv("mutation_data_4_ML.csv")


# Step 2: Prepare data for modeling - separate X from y and split train, validation and test sets

# Step 2A
X = ds.drop(['Target'],axis=1)
y = ds[['Target']] 

# Step 2B: Splitting the dataset into the Training and Testing set.

X_large, X_test, y_large, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42,stratify=y,shuffle = True)

# step 2C: Create dummy variables for categorical variables (object data types)

X_train = pd.get_dummies(X_train, columns=["Type","Gene"],drop_first=True)
X_test = pd.get_dummies(X_test, columns=["Type","Gene"],drop_first=True)
X_val = pd.get_dummies(X_val, columns=["Type","Gene"],drop_first=True)

# Step 2D: Normalize all numerical features 

sc=StandardScaler()
temp = sc.fit(X_train[['Number_of_nucleotide_changes',
 'C_score',
 'phred_like_score',
 'MSRV_HVAR_score',
 'MSRV_HGMD_score',
 'CONDEL_db-version_05_score',
 'Uniprot_aapos',
 'SiPhy_score',
 'SIFT_Ensembl_66_score',
 'CanDrA_melanoma_v1.0_score',
 'rate_A',
 'VEST_v3.0_score',
 'CanDrA_breast_v1.0_score',
 'CanDrA_lung_v1.0_score',
 'PROVEAN_v1.1.3_score',
 'PolyPhen-2_v2.2.2_probability',
 'Mutation_Assessor_release_2_score',
 'FATHMM_missense_v2.3_score',
 'FATHMM_cancer_v2.3_score',
 'CHASM_melanoma_v1.0.7_score',
 'CHASM_lung_v1.0.7_score',
 'CHASM_breast_v1.0.7_score',
 'rate_C',
 'rate_G',
 'rate_T',
 'FATHMM_score',
 'Polyphen2_HVAR_score']])
X_train[['Number_of_nucleotide_changes',
 'C_score',
 'phred_like_score',
 'MSRV_HVAR_score',
 'MSRV_HGMD_score',
 'CONDEL_db-version_05_score',
 'Uniprot_aapos',
 'SiPhy_score',
 'SIFT_Ensembl_66_score',
 'CanDrA_melanoma_v1.0_score',
 'rate_A',
 'VEST_v3.0_score',
 'CanDrA_breast_v1.0_score',
 'CanDrA_lung_v1.0_score',
 'PROVEAN_v1.1.3_score',
 'PolyPhen-2_v2.2.2_probability',
 'Mutation_Assessor_release_2_score',
 'FATHMM_missense_v2.3_score',
 'FATHMM_cancer_v2.3_score',
 'CHASM_melanoma_v1.0.7_score',
 'CHASM_lung_v1.0.7_score',
 'CHASM_breast_v1.0.7_score',
 'rate_C',
 'rate_G',
 'rate_T',
 'FATHMM_score',
 'Polyphen2_HVAR_score']] = temp.transform(X_train[['Number_of_nucleotide_changes',
 'C_score',
 'phred_like_score',
 'MSRV_HVAR_score',
 'MSRV_HGMD_score',
 'CONDEL_db-version_05_score',
 'Uniprot_aapos',
 'SiPhy_score',
 'SIFT_Ensembl_66_score',
 'CanDrA_melanoma_v1.0_score',
 'rate_A',
 'VEST_v3.0_score',
 'CanDrA_breast_v1.0_score',
 'CanDrA_lung_v1.0_score',
 'PROVEAN_v1.1.3_score',
 'PolyPhen-2_v2.2.2_probability',
 'Mutation_Assessor_release_2_score',
 'FATHMM_missense_v2.3_score',
 'FATHMM_cancer_v2.3_score',
 'CHASM_melanoma_v1.0.7_score',
 'CHASM_lung_v1.0.7_score',
 'CHASM_breast_v1.0.7_score',
 'rate_C',
 'rate_G',
 'rate_T',
 'FATHMM_score',
 'Polyphen2_HVAR_score']])


X_test[['Number_of_nucleotide_changes',
 'C_score',
 'phred_like_score',
 'MSRV_HVAR_score',
 'MSRV_HGMD_score',
 'CONDEL_db-version_05_score',
 'Uniprot_aapos',
 'SiPhy_score',
 'SIFT_Ensembl_66_score',
 'CanDrA_melanoma_v1.0_score',
 'rate_A',
 'VEST_v3.0_score',
 'CanDrA_breast_v1.0_score',
 'CanDrA_lung_v1.0_score',
 'PROVEAN_v1.1.3_score',
 'PolyPhen-2_v2.2.2_probability',
 'Mutation_Assessor_release_2_score',
 'FATHMM_missense_v2.3_score',
 'FATHMM_cancer_v2.3_score',
 'CHASM_melanoma_v1.0.7_score',
 'CHASM_lung_v1.0.7_score',
 'CHASM_breast_v1.0.7_score',
 'rate_C',
 'rate_G',
 'rate_T',
 'FATHMM_score',
 'Polyphen2_HVAR_score']] = temp.transform(X_test[['Number_of_nucleotide_changes',
 'C_score',
 'phred_like_score',
 'MSRV_HVAR_score',
 'MSRV_HGMD_score',
 'CONDEL_db-version_05_score',
 'Uniprot_aapos',
 'SiPhy_score',
 'SIFT_Ensembl_66_score',
 'CanDrA_melanoma_v1.0_score',
 'rate_A',
 'VEST_v3.0_score',
 'CanDrA_breast_v1.0_score',
 'CanDrA_lung_v1.0_score',
 'PROVEAN_v1.1.3_score',
 'PolyPhen-2_v2.2.2_probability',
 'Mutation_Assessor_release_2_score',
 'FATHMM_missense_v2.3_score',
 'FATHMM_cancer_v2.3_score',
 'CHASM_melanoma_v1.0.7_score',
 'CHASM_lung_v1.0.7_score',
 'CHASM_breast_v1.0.7_score',
 'rate_C',
 'rate_G',
 'rate_T',
 'FATHMM_score',
 'Polyphen2_HVAR_score']])
X_val[['Number_of_nucleotide_changes',
 'C_score',
 'phred_like_score',
 'MSRV_HVAR_score',
 'MSRV_HGMD_score',
 'CONDEL_db-version_05_score',
 'Uniprot_aapos',
 'SiPhy_score',
 'SIFT_Ensembl_66_score',
 'CanDrA_melanoma_v1.0_score',
 'rate_A',
 'VEST_v3.0_score',
 'CanDrA_breast_v1.0_score',
 'CanDrA_lung_v1.0_score',
 'PROVEAN_v1.1.3_score',
 'PolyPhen-2_v2.2.2_probability',
 'Mutation_Assessor_release_2_score',
 'FATHMM_missense_v2.3_score',
 'FATHMM_cancer_v2.3_score',
 'CHASM_melanoma_v1.0.7_score',
 'CHASM_lung_v1.0.7_score',
 'CHASM_breast_v1.0.7_score',
 'rate_C',
 'rate_G',
 'rate_T',
 'FATHMM_score',
 'Polyphen2_HVAR_score']] = temp.transform(X_val[['Number_of_nucleotide_changes',
 'C_score',
 'phred_like_score',
 'MSRV_HVAR_score',
 'MSRV_HGMD_score',
 'CONDEL_db-version_05_score',
 'Uniprot_aapos',
 'SiPhy_score',
 'SIFT_Ensembl_66_score',
 'CanDrA_melanoma_v1.0_score',
 'rate_A',
 'VEST_v3.0_score',
 'CanDrA_breast_v1.0_score',
 'CanDrA_lung_v1.0_score',
 'PROVEAN_v1.1.3_score',
 'PolyPhen-2_v2.2.2_probability',
 'Mutation_Assessor_release_2_score',
 'FATHMM_missense_v2.3_score',
 'FATHMM_cancer_v2.3_score',
 'CHASM_melanoma_v1.0.7_score',
 'CHASM_lung_v1.0.7_score',
 'CHASM_breast_v1.0.7_score',
 'rate_C',
 'rate_G',
 'rate_T',
 'FATHMM_score',
 'Polyphen2_HVAR_score']])

# Model Evaluation Criteria

# Model can make wrong predictions as:

# - Predicting a mutation is driver whereas it is a passenger (case 1)
# - Predicting a mutation is passenger whereas it is a driver (case 2)

# Which case is more important?

# case 2 is more severe than case 1

#How to reduce this loss i.e need to reduce False Negative?

# - want Recall to be maximized, greater the Recall higher the chances of minimizing false Negative. Hence, the focus should be on increasing Recall or minimizing the false Negative or in other words identifying the True Positive(i.e. Class 1)

# Step 3: Create function for evaluation metrics - confusion matrix and other metrics

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

# Step 4 : NeuralNets Model Build

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(21)
random.seed(21)
tf.random.set_seed(21)

# Initializing the ANN
classifier = Sequential()
# Adding the input layer with 64 neurons with relu as activation function with input of 32 variables
classifier.add(Dense(activation = 'relu', input_dim = 32, units=64))
#Adding 1st hidden layer with 32 neurons
classifier.add(Dense(32, activation='relu'))
# Adding the output layer
# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
# We use the sigmoid because we want probability outcomes
classifier.add(Dense(1, activation = 'sigmoid'))

# Compile the model with adam optimizer and binary cross entropy as loss with accuracy as metrics
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
classifier.summary()


history=classifier.fit(X_train, y_train,
          validation_data=(X_val,y_val),
          epochs=80,
          batch_size=32)

# Step 5: Loss function

# Capturing learning history per epoch
hist  = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

# Plotting accuracy at different epochs
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(("train" , "valid") , loc =0)

#Printing results
results = classifier.evaluate(X_test, y_test)


# Step 6: #Calculating the confusion matrix
y_pred1=classifier.predict(X_val)

#Let's predict using default threshold
y_pred1 = (y_pred1 > 0.5)
cm2=confusion_matrix(y_val, y_pred1)
labels = ['TN','FP','FN','TP']
categories = [ 'Passenger','Driver']
make_confusion_matrix(cm2,
                      group_names=labels,
                      categories=categories,
                      cmap='Blues')

def create_model():
      #Initializing the neural network
      model = Sequential()
      #Adding the input layer with 64 neurons and relu as activation function
      model.add(Dense(64,activation='relu',input_dim = X_train.shape[1]))
      # Adding the first hidden layer with 32 neurons with relu as activation functions
      model.add(Dense(32,activation='relu'))
      # Adding the output layer
      model.add(Dense(1, activation = 'sigmoid'))
      #Compiling the ANN with Adam optimizer
      optimizer = tf.keras.optimizers.Adam(0.001)
      # Complining the model with binary cross entropy as loss function and accuracy as metrics
      model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
      return model


model=create_model()
model.summary()

history = model.fit(X_train,y_train,batch_size=32,validation_data=(X_val,y_val),epochs=100,verbose=1)

#Plotting Train Loss vs Validation Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Defining Early stopping
es_cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)
model_e=create_model()
#Fitting the ANN with batch_size = 32 and 100 epochs
history_e = model_e.fit(X_train,y_train,batch_size=32,epochs=100,verbose=1,validation_data=(X_val,y_val),callbacks=[es_cb])

#Plotting Train Loss vs Validation Loss
plt.plot(history_e.history['loss'])
plt.plot(history_e.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# predict probabilities
yhat = model_e.predict(X_train)
# keep probabilities for the positive outcome only
yhat = yhat[:, 0]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(y_train, yhat)
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()

#Predicting the results using best as a threshold
y_pred_e=model_e.predict(X_val)
y_pred_e = (y_pred_e > thresholds[ix])
y_pred_e


#Accuracy as per the classification report
cr=metrics.classification_report(y_val,y_pred_e)
print(cr)

cm1=confusion_matrix(y_val, y_pred_e)
labels = ['True Negative','False Positive','False Negative','True Positive']
categories = [ 'Not_Exited','Exited']
make_confusion_matrix(cm1,
                      group_names=labels,
                      categories=categories,
                      cmap='Blues')


# adding Dropout layer to improve model performance

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)


model_3 = Sequential()
#Adding the input layer with 32 neurons and relu as activation function
model_3.add(Dense(32,activation='relu',input_dim = X_train.shape[1]))
# Adding dropout with ratio of 0.2
model_3.add(Dropout(0.2))
# Adding the first hidden layer with 16 neurons with relu as activation functions
model_3.add(Dense(16,activation='relu'))
# Adding dropout with ratio of 0.1
model_3.add(Dropout(0.1))
# Adding the second hidden layer with 8 neurons with relu as activation functions
model_3.add(Dense(8,activation='relu'))
# Adding the output layer
model_3.add(Dense(1, activation = 'sigmoid'))

# Summary of the model
model_3.summary()

#Compiling the ANN with Adam optimizer
optimizer = tf.keras.optimizers.Adam(0.001)
# Complining the model with binary cross entropy as loss function and accuracy as metrics
model_3.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])


#Fitting the ANN with batch_size = 32 and 100 epochs
history_3 = model_3.fit(X_train,y_train,batch_size=32,epochs=100,verbose=1,validation_data=(X_val,y_val),callbacks=[es_cb])

#visualize loss function 
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#Predicting the results using best as a threshold
y_pred_e=model_3.predict(X_val)
y_pred_3 = (y_pred_e > thresholds[ix])
y_pred_3

#Accuracy as per the classification report
cr=metrics.classification_report(y_val,y_pred_3)
print(cr)


#Calculating the confusion matrix

cm1=confusion_matrix(y_val,y_pred_3)
labels = ['True Negative','False Positive','False Negative','True Positive']
categories = [ 'Passenger','Driver']
make_confusion_matrix(cm1,
                      group_names=labels,
                      categories=categories,
                      cmap='Blues')


# Use Grid Search Hyperparameter tuning

#Some important parameters to look out for while optimizing neural networks are:

#-Type of architecture

#-Number of Layers

#-Number of Neurons in a layer

#-Regularization parameters

#-Learning Rate

#-Type of optimization / backpropagation technique to use

#-Dropout rate

#-Weight sharing

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

# @title
def create_model_v2(dropout_rate=0.1,lr=0.001,layer_1=64,layer_2=32):
    np.random.seed(1337)
    #Initializing the neural network
    model = Sequential()
    # This adds the input layer (by specifying input dimension)
    model.add(Dense(layer_1,activation='relu',input_dim = X_train.shape[1]))
    #Adding dropout layer
    model.add(Dropout(0.5))
    # # Adding the hidden layer
    # Notice that we do not need to specify input dim.
    model.add(Dense(layer_2,activation='relu'))
    # # Adding the output layer
    # Notice that we do not need to specify input dim.
    # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
    # We use the sigmoid because we want probability outcomes
    model.add(Dense(1, activation='sigmoid'))

    # Adding Adam initializer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #compile model
    model.compile(optimizer = optimizer,loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


keras_estimator = KerasClassifier(build_fn=create_model_v2, verbose=1)


# define the grid search parameters
param_grid = {
    'batch_size':[40, 64, 128],
    "lr":[0.01,0.001,0.1]}


kfold_splits = 3
# Applying GridSearchCV
grid = GridSearchCV(estimator=keras_estimator,
                    verbose=1,
                    cv=kfold_splits,
                    param_grid=param_grid,n_jobs=-1)

## Fitting Grid model
grid_result = grid.fit(X_train, y_train,validation_data = (X_val,y_val),verbose=1)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Printing mean
means = grid_result.cv_results_['mean_test_score']
# Printing standard deviation
stds = grid_result.cv_results_['std_test_score']
# Printing best parameters
params = grid_result.cv_results_['params']

# recommended parameters 

#Result of Grid Search

#{'batch_size': 40, 'learning_rate":0.01}

#Heuristic for Hyperparameters

#optimizer="adam", layer1_units=64, layer2_units = 32

# Creating the model
estimator_v2=create_model_v2(lr=grid_result.best_params_['lr'])
# Printing model summary
estimator_v2.summary()

## Fitting the model
history_h=estimator_v2.fit(X_train, y_train, epochs=100, batch_size = grid_result.best_params_['batch_size'], verbose=1,validation_data=(X_val,y_val))

# visualize loss function 
N =100
plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history_h.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history_h.history["val_loss"], label="val_loss")

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="center")
plt.show()

# pridiction
y_pred_h = estimator_v2.predict(X_val)
y_pred_h = (y_pred_h > thresholds[ix])
print(y_pred_h)
