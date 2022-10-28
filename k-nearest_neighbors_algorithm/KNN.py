import numpy as np
from sklearn.datasets import load_iris
from collections import Counter

def kNN_Classifier():
    print('\nk-NN Classifier')
    

    dataset = load_iris()

    X = dataset.data
    y = dataset.target
    y = y.reshape([len(y),1])
    X = np.concatenate((X,y),axis=1)

    np.random.shuffle(X)

    train_data = X[0:120]
    test_data = X[120:150]

    for k in range(3,11):
        success = 0
        for n in range(len(test_data)):
            distances = np.sqrt(np.sum((train_data[: ,:4] - test_data[n,:4])**2, axis=1))
            near_labels = train_data[np.argpartition(distances,np.arange(0, k, 1))[0:k],4] 
            ans_counts = Counter(near_labels)

            if ans_counts.most_common(1)[0][0] == test_data[n][4]:
                success += 1

        print('accuracy for k =',k,':',round((success/len(test_data))*100),'%')

kNN_Classifier()
