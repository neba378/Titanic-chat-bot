import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("data/titanic.csv")

def plot_age_histogram():
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"].dropna(), bins=20, kde=True)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Histogram of Passenger Ages")
    plt.savefig("plots/age_histogram.png")
    return "plots/age_histogram.png"
