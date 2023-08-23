import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "font.size": 13,
}
rcParams.update(config)

methods = ["Adaboost", "xgboost", "SVM", "MultiLayerPerceptron", "Logistic", "RandomForest"]

pretty_methods = {"Adaboost": "ABRF", "xgboost": "XGB", "MultiLayerPerceptron": "MP",
                  "Logistic": "LR", "RandomForest": "RF", "SVM": "SVM"}
for method in methods:
    xy = pd.read_csv("../Results/FeatureSizes/" + method + ".csv", sep=" ", header=None)
    x = list(xy[xy.columns[0]])
    y = list(xy[xy.columns[1]])
    plt.plot(x, y, label=pretty_methods[method], marker="o")

plt.legend(loc="best")
plt.xlabel("# of features")
plt.ylabel("Average AUC")
plt.savefig("../Results/FeatureSizes/FeatureComparison.png")
