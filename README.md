# Energy-Dataset-Forward-Feed-Neural-Network

Numpy, pandas and sklearn were used to develop a neural network model for an energy dataset. 
The two continuous response variables predicted were heating load and cooling load. Eight inputs were
used to predict these response variables. 
  The data predictor variables were scaled between 0 and 1. A train test split of 60/40 was used. 
  A neural network architecture was built using two hidden layers, with 30 nodes each, and a learning rate of 0.001.
  The RMSE and R2 score was used as a measure of determining the neural networks performance. 
  The mean and standard deviation of the RMSE and R2 score across 10 experiments for the neural network model was calculated.
  This was compared with mean and standard deviation of the RMSE and R2 score using a linear regression model.
  The neural network model had a higher R2 score and lower RMSE, performing better than the linear regression model
  when solely considering these metrics. 
