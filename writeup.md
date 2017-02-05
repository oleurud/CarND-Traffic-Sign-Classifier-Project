####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/oleurud/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images of each label.

![alt text][figure.jpg]

We can see the dataset is not well distributed, which will make it difficult to learn our system.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I did a shuffle of the training data, as usual, to improve the learning.

Then, after read the paper of the New York University, I tried to use grayscale images, but the result were good.

As a last step, I normalized the image data with nice results. The test data of the dataset was already normalized and I discover it thanks to this process.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

I did it using the function train_test_split of the sklearn library, using the 20% of the initial training data to create the validation set.

After that:

* The size of training set is 31367
* The size of test set is 12630
* The size of validation set is 7842

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution        	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution        	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten   	      	| outputs 400                    				|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout				| keep_prob of 0.5 in training step				|
| Softmax				|           									|
| Adam Optimizer		|												|
 
Basically is the LeNet with a final Dropout layer.

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

Together with the step before, I tried a lot of things:

* I tried to use other optimizer, but no one works better (the mayority no works... my fault, for sure)
* I tried to play with parameters like learning rate, the mean or standard desviation in covnets, ... 
* I tried to remove the max pooling (because they lose information) modifying the architecture. The best try was using 4 convolutional (with 2x2 stride for parameters control) and 2 full conected, but the results were the same and the performance worst (2 times more)
* I played a lot with dropout, trying to reduce the overfitting. The best result was one unique layer at the final with a keep probability of 0.5. With this solution, the validation decrease from 98% from 95%, but the test acurracy increase a litte, from 85% to 88%

It was a bit frustrating, because I spend a lot of time reading and trying other ideas, but finally, the LeNet was the best option.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.95 
* test set accuracy of 0.88

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet
* What were some problems with the initial architecture?
The default LeNet works fine with a 95% of validation set accuracy. But it has overfitting, the validation accuracy was a 85% 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
How I have explained before, I tried a lot of things: other architectures, diferent dropouts layers with diferents parameters, remove the max pooling, grayscale images, ... The first success was implement the image normalization from -1 to 1 values
* Which parameters were tuned? How were they adjusted and why?
Learning rate, keep probabilities, epochs, mean, standard desviation... I tried to use the ideas of the classes, and based in trial and error, starting in the recomended vales, doing changes and looking the validation results
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The two things that works for me were the final dropout and the images normalization

If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
I am not sure, but I think that my results are not good enough... and 85% it is not safe
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Not provide a evidence

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs which I have worked. The mayority are fine, maybe with a smalls bad details:

![Speed limit (30km/h)][new-images/1.jpg]
Has a little cut in the bottom
![Speed limit (50km/h)][new-images/2.jpg]
![Speed limit (70km/h)][new-images/4.jpg]
It's a little distorted
![Yield][new-images/13.jpg]
![Stop][new-images/14.jpg]
![No entry][new-images/17.jpg]
Has a little cut in the bottom
![Road work][new-images/25.jpg]
![Wild animals crossing][new-images/31.jpg]
![Keep right][new-images/38.jpg]
![Roundabout mandatory][new-images/40.jpg]
For me the unique with real problems, beacause the sign has parts of a tree in front


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        			|
|:---------------------:|:-------------------------------------:|
| Speed limit (30km/h)  | Speed limit (80km/h)					| 
| Road work     		| Road work								|
| Stop					| Stop									|
| Yield	      		    | Yield     				 			|
| Speed limit (70km/h)	| Speed limit (70km/h) 					|
| Roundabout mandatory	| Roundabout mandatory					|
| Wild animals crossing	| Wild animals crossing					|
| Keep right			| Keep right							|
| Speed limit (50km/h)	| Ahead only							|
| No entry				| No entry								|


In this case, the model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. More or less, the results are the same with the validation set and tthis new real images.

The most of the time, the model has errors with the speed limit signs. I dont understand why, because the learning was with a lot of examples of this types of signs (I espect that the model try to respond more times this types)

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the 1st image, Speed limit (30km/h). The model has a lot of doubts with similar signs, especialy the 3 first, but no one is the correct.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .18         			| Speed limit (80km/h)							| 
| .15     				| Speed limit (50km/h)							|
| .15					| Speed limit (60km/h)							|
| .5	      			| Speed limit (100km/h)			 				|
| .2				    | Speed limit (20km/h)							|


For the 2nd image, Road work. The model is relatively sure because one has a 24% and the rest very low percentaje.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .24         			| Road work  									| 
| .04     				| Speed limit (80km/h)							|
| .05					| Wild animals crossing							|
| .01	      			| Bicycles crossing				 				|
| .00				    | Speed limit (30km/h) 							|


For the 3rd image, Stop. The model has a lot of doubts with circular signs. It success, but it is not sure
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .13         			| Stop         									| 
| .08     				| No passing 									|
| .06					| No vehicles									|
| .06	      			| Speed limit (60km/h)			 				|
| .03				    | Speed limit (80km/h)      					|


For the 4th image, Yield. The model is relatively sure that this is a yield sign (51%)
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| Yield        									| 
| .22     				| No vehicles 		    						|
| .17					| Bumpy road									|
| .06	      			| No passing					 				|
| .01				    | Speed limit (60km/h) 							|


For the 5th image, Speed limit (70km/h). The model has a lot of doubts with speed limit signs. It success, but it is not sure
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .11         			| Speed limit (70km/h)							| 
| .05     				| Speed limit (50km/h)							|
| .03					| Speed limit (100km/h							|
| .03	      			| Speed limit (30km/h)			 				|
| .00				    | Speed limit (120km/h)							|


For the 6th image, Roundabout mandatory. The model has 2 options highlighted and the rest with very low percentaje. It success, but it is not sure
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .08         			| Roundabout mandatory   						| 
| .06     				| End of all speed and passing limits			|
| .00					| Go straight or left							|
| .00	      			| Speed limit (20km/h)			 				|
| .00				    | Speed limit (30km/h)      					|


For the 7th image, Wild animals crossing. The model has a lot of doubts with triangular signs. Again, it success, but it is not sure
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .18         			| Wild animals crossing  						| 
| .13     				| Double curve 									|
| .08					| Road narrows on the right						|
| .00	      			| Slippery road					    			|
| .00				    | Pedestrians      				    			|


For the 8th image, Keep right. The model is relatively sure that this is a "Keep right" sign (77%)
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .77         			| Keep right   									| 
| .22     				| Turn left ahead								|
| .08					| Yield										    |
| .00	      			| Priority road					 				|
| .00				    | Go straight or right      					|


For the 9th image, Speed limit (50km/h). I can not understand what is happening here. The learning must be improved
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .07         			| Ahead only   									| 
| .05     				| Roundabout mandatory 							|
| .03					| No passing for vehicles over 3.5 metric tons	|
| .02	      			| Turn right ahead					 	    	|
| .00				    | End of no passing by vehicles over 3.5 		|


For the 10th image, No entry. The model is relatively sure that this is a "no entry" sign (22%) because the rest of the options has a very low percentaje
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .22         			| No entry   									| 
| .03     				| Speed limit (30km/h)							|
| .01					| Speed limit (20km/h)							|
| .00	      			| Wild animals crossing			 				|
| .00				    | Bicycles crossing    							|
