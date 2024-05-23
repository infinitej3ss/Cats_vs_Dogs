import os
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def plot_images(images, labels, preds):
    plt.figure(figsize = (10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(100, 100), cmap = 'grey')
        plt.title(f'Label: {labels[i]}\nPred: {preds[i]}')
        plt.axis('off')
    plt.show()

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img = img.convert('L')
            images.append(np.array(img).flatten())
            labels.append(0 if 'cat' in filename else 1)
    return np.array(images), np.array(labels)

path_cats = '/[os.getcwd()]/Cats and Dogs (Output)/Cats'
path_dogs = '/[os.getcwd()]/Cats and Dogs (Output)/Dogs'

images_cats, labels_cats = load_images(path_cats)
images_dogs, labels_dogs = load_images(path_dogs)


X = np.vstack((images_cats, images_dogs))
y = np.concatenate((labels_cats, labels_dogs))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state = 2)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plot_images(X_test, y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
