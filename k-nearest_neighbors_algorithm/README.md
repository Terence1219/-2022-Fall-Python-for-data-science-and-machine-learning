# -2022-Fall-Python-for-data-science-and-machine-learning
## KNN algorithm with numpy
KNN algorithm choose K nearest neighbor in the space to determine the value of the input.  
It's a kind of Supervised learning  
  
### 0.import module
```python
import numpy as np
from sklearn.datasets import load_iris
from collections import Counter
```
### 1.load data
```python
  
dataset = load_iris()

    X = dataset.data
    y = dataset.target
    y = y.reshape([len(y),1])
    X = np.concatenate((X,y),axis=1)

    np.random.shuffle(X)

    train_data = X[0:120]
    test_data = X[120:150]
```
First, we import Iris dataset from sklearn.  
We can concatenate data and target(last column), and shuffle them together.  
Finally, we split the data into training(80%) and testing(20%).

### 2.train model and test it
```python
for k in range(3,11):
        success = 0
        for n in range(len(test_data)):
            distances = np.sqrt(np.sum((train_data[: ,:4] - test_data[n,:4])**2, axis=1))
            near_labels = train_data[np.argpartition(distances,np.arange(0, k, 1))[0:k],4] 
            ans_counts = Counter(near_labels)

            if ans_counts.most_common(1)[0][0] == test_data[n][4]:
                success += 1

        print('accuracy for k =',k,':',round((success/len(test_data))*100),'%')
```
The value of k we set is ranging from 3 to 11  
```python
distances = np.sqrt(np.sum((train_data[: ,:4] - test_data[n,:4])**2, axis=1))
```
First, we calculate the Euclidean distance between n-th training data(anchor) and other training data.  
```python
near_labels = train_data[np.argpartition(distances,np.arange(0, k, 1))[0:k],4] 
```
We choose k labels according to the least k elements in distance list  
```python
ans_counts = Counter(near_labels)
if ans_counts.most_common(1)[0][0] == test_data[n][4]:
   success += 1
```
We count the number of labels and check whether the label which accounts for the most is equivalent to the label of n-th training data  
```python
print('accuracy for k =',k,':',round((success/len(test_data))*100),'%')

'''
accuracy for k = 3 : 93 %
accuracy for k = 4 : 93 %
accuracy for k = 5 : 100 %
accuracy for k = 6 : 97 %
accuracy for k = 7 : 100 %
accuracy for k = 8 : 100 %
accuracy for k = 9 : 100 %
accuracy for k = 10 : 100 %
'''
```
Finally, we print out the accuracy of each k number.

## Conclusion
If the intra-class data is close and the inter-class data is separated, the method of KNN alogorithm can have high performance.  

The followings plot uses the features of the Iris dataset.
You can observe the relationship between different types of irises and features. 
![First three PCA directions](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dataset_001.png "First three PCA directions")
![Sepal Length and Sepal Width in Iris dataset](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dataset_002.png "Sepal Length and Sepal Width in Iris dataset")
## Reference
[Code](https://ithelp.ithome.com.tw/articles/10210788)  
[Images](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py)
