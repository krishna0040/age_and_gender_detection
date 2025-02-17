# age_and_gender_detection
Tensorflow model which predicts the age and gender of people in an image.
The model downloads the UTK-facenew datset from kaggle, which as about 10000 labeled images. 

Then resize the data to a 70x70 and a gray scale image, which is easy for the model to train on and it also take about 1/3 space as that of an RGB image, which was causing memory issues, and it was not required for this task. 

Then check if the data is unbalanced, which it was not in terms of gender, but for age, the data was unbalanced.
Then scale the image and define a keras model using tensorflow backend, which uses convulational neural network.

Define 2 output for the same model, one for age and the other for gender. The age output uses mean absolute error as its metrics, while gender output uses accuracy.
After training the model and using best weights for the model,** we got an accuracy of 90% on the gender prediction and a mae of age about 4 years on the test data.** 
Thus the model is memory efficient (uses memory less than 12Gb ram overall) and at the same time also produces good results. 

The age_and_gender_detection_gui detects the face using yolov8n-face-lindevs.pt, as it is faster and better than haarcascade, which i was previously using. Thee gui takes and image as input and draws a bounding box around the face and shows the age and gender of the person.

The live_detection_gui open a livecam and detects in real time.

The video_detection_gui uses the video as input and detects the age and gender. For this, and the live_detection gui, a dictionary is made and a unique id is given to each. This is done to prevent the model from predicting the same face again and again.
