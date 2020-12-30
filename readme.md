#### Model

```
This experiment uses LSTM framework with self-attention mechanism to learn sequence classification. 

I checked several papers, a lot of them use CNN for multi-variate sequences classification, but since this gesture dataset has only three dimensions, LSTM can work well. The attention mechanism is used to help capture the long-term dependency.
```

#### run

```
cd to the root folder and run python3 classifcation.py
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

#### visualization

```
I visualized the training accuracy and recall for each epoch. 

You can run classification.py to get it, or you can check it directly through 'visualization.png' in the root folder.
```

![image-20201230013355626](fig/image-20201230013355626.png)

