#### Model

```
This experiment uses LSTM framework with self-attention mechanism to learn sequence classification. 

I checked several papers, a lot of them use CNN for multi-variate sequences classification, but since this gesture dataset has only three dimensions, LSTM can work well. The attention mechanism is used to help capture the long-term dependency.
```

#### run

```
cd to the root folder and run python3 classifcation.py
```
#### data

```
data source: http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip
data discription: http://timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryAll

On the top level, each .rar file includes the gesture samples collected from one user on one day.
The .rar files are named as U$userIndex ($dayIndex).rar, where $userIndex is the index of the participant from 1 to 8, and $dayIndex is the index of the day from 1 to 7.

Inside each .rar file, there are .txt files recording the time series of acceleration of each gesture.
The .txt files are named as [somePrefix]$gestureIndex-$repeatIndex.txt, where $gestureIndex is the index of the gesture as in the 8-gesture vocabulary, and $repeatIndex is the index of the repetition of the same gesture pattern from 1 to 10.

In each .txt file, the first column is the x-axis acceleration, the second y-axis acceleration, and the third z-axis acceleration.
The unit of the acceleration data is G, or acceleration of gravity.
```

#### Requirements

```
python3
torch >= 1.7.1
sklearn >= 0.23
patoolib
rarfile
zipfile
matplotlib
```

#### Results

```
I used 20% data for training and 80% for testing, after 100 epoch, the model can achieve 98% accuracy and 98% recall. 

			precision    recall  f1-score   support

           0       0.99      0.96      0.98       466
           1       0.99      0.99      0.99       437
           2       0.97      0.97      0.97       456
           3       0.99      0.98      0.99       444
           4       0.99      1.00      0.99       453
           5       0.97      0.97      0.97       452
           6       0.98      0.99      0.99       434
           7       0.98      1.00      0.99       443

    accuracy                           0.98      3585
   macro avg       0.98      0.98      0.98      3585
weighted avg       0.98      0.98      0.98      3585

```
