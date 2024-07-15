from keras.src.metrics.accuracy_metrics import accuracy
from numpy import shape
import pandas as pd

# import matplotlib as mlp
import matplotlib.pyplot as plt

# import seaborn as sns
from sklearn.model_selection import train_test_split
from keras import Input, Sequential, layers

# from keras.layers import Dense


def logAttr(arg):
    print(f"Attributes of {arg}")
    attrs = [attr for attr in dir(arg) if not attr.startswith("__")]
    for attr in attrs:
        print(attr)


dataset = pd.read_csv("diabetes.csv")
# print(dataset.describe(include="all"))
# print(dataset.corr())

# print(dataset.columns)
fig, ax = plt.subplots()
ax.bar(dataset.index, dataset["Age"])

x_variables = dataset.iloc[:, 0:8]
y_variables = dataset.iloc[:, 8]
# print(x_variables)
# print(y_variables)


x_train, x_test, y_train, y_test = train_test_split(
    x_variables, y_variables, test_size=0.2, random_state=10
)


model = Sequential(
    [
        layers.Dense(15, activation="relu"),
        layers.Dense(12, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_variables, y_variables, epochs=95, batch_size=25)
_, accuracy = model.evaluate(x_variables, y_variables)
print("accuracy: %.2f" % (accuracy * 100))
model.summary()
# ax.set_xticks(dataset.index)
# ax.set_xticklabels(dataset.index, rotation=60, horizontalalignment="right")
# plt.show()
