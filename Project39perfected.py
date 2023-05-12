import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GUI(tk.Frame):
    def __init__(self):
        super().__init__()
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.open_file_button = tk.Button(
            self, text="Άνοιγμα Αρχείου", command=self.open_file
        )
        self.open_file_button.pack(side="top")

        self.statistics_label = tk.Button(
            self, text="Εμφάνιση Στατιστικών", command=self.show_statistics
        )
        self.statistics_label.pack(side="top")

        self.run_algorithm_button = tk.Button(
            self, text="Εκτέλεση Αλγόριθμου", command=self.run_algorithm
        )
        self.run_algorithm_button.pack(side="top")

    def open_file(self):
        file_path = filedialog.askopenfilename()
        self.data = pd.read_csv(file_path, header=None)
        self.data = self.data.replace("M", 1)
        self.data = self.data.replace("B", 0)

    def show_statistics(self):
        mean = self.data.iloc[:, 1].mean()
        median = self.data.iloc[:, 1].median()
        low_mean = self.data.iloc[:, 1].mean() - self.data.iloc[:, 1].std()
        high_mean = self.data.iloc[:, 1].mean() + self.data.iloc[:, 1].std()
        standard_deviation = self.data.iloc[:, 1].std()
        variance = self.data.iloc[:, 1].var()
        plt.bar(["Mean", "Variance"], [mean, variance])
        plt.xlabel("Statistics")
        plt.ylabel("Value")
        plt.title("Mean and Variance for Diagnosis")
        plt.show()

    def run_algorithm(self):
        X = self.data.iloc[:, 2:].values
        y = self.data.iloc[:, 1].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        accuracy = str(accuracy_score(y_test, y_pred) * 100)[0:5] + "%"
        tk.messagebox.showinfo(title="Ευστοχία", message=accuracy)


gui = GUI()
gui.mainloop()
