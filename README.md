## Developing-a-Neural-Network-Regression-Model
## AIM
To develop a neural network regression model to predict a student's final marks based on the number of hours they study

## THEORY
In educational institutions, understanding the direct impact of study habits on academic performance is crucial. The goal of this project is to build a predictive system that takes a single continuous feature—the number of hours a student spends studying—and outputs a continuous numerical prediction of their expected final exam score. Because the target variable (marks) is a continuous numerical value rather than a discrete category, this challenge is classified as a regression problem

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
## STEP 1:
Create your dataset in a Google sheet with one numeric input and one numeric output.

## STEP 2:
Split the dataset into training and testing

## STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.

## STEP 4:
Build the Neural Network Model and compile the model.

## STEP 5:
Train the model with the training data.

## STEP 6:
Plot the performance plot

## STEP 7:
Evaluate the model with the testing data.

## STEP 8:
Use the trained model to predict for a new input value .

## PROGRAM
```
Name: Yogesh G
Register Number:212225100061
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("student_study_hours_regression.csv")

print(df)
X = df[["Study_Hours"]].values
y = df[["Exam_Marks"]].values

print("Input:")
print(X)

print("Output:")
print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(X_train.shape)
print(y_train.shape)
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
torch.manual_seed(33)

model = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)
epochs = 1000
loss_list = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
model.eval()

with torch.no_grad():
    predictions = model(X_test).numpy()

print("\nModel Evaluation")
print("MSE:", mean_squared_error(y_test.numpy(), predictions))
print("RMSE:", mean_squared_error(y_test.numpy(), predictions) ** 0.5)
print("MAE:", mean_absolute_error(y_test.numpy(), predictions))
print("R2 Score:", r2_score(y_test.numpy(), predictions))
plt.figure(figsize=(8, 4))
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Student Marks Prediction - Training Loss")
plt.show()
new_student = [[9]]

new_student_scaled = scaler.transform(new_student)

new_student_tensor = torch.tensor(
    new_student_scaled,
    dtype=torch.float32
)

with torch.no_grad():
    predicted_marks = model(new_student_tensor).item()

print(f"Predicted Exam Marks for 9 Study Hours: {predicted_marks:.2f}")
```
## Dataset Information
Include screenshot of the generated data

## OUTPUT


## Dataset Information
<img width="708" height="477" alt="image" src="https://github.com/user-attachments/assets/f3fd9a61-b4a6-42f5-bcaa-5086b265f369" />


## Training samples and testing samples

<img width="512" height="212" alt="image" src="https://github.com/user-attachments/assets/fcf23916-f33a-4982-8602-7bb5520e521d" />

## Loss list

<img width="660" height="580" alt="image" src="https://github.com/user-attachments/assets/742980de-9ef6-4809-b2d1-657ac8921f27" />

## Model evaluation


<img width="850" height="416" alt="image" src="https://github.com/user-attachments/assets/fb364bef-e1aa-4335-8c76-e173eec78d23" />

## Plot Diagram

<img width="787" height="437" alt="image" src="https://github.com/user-attachments/assets/edacff45-f4f2-44c9-bda0-4f7a554b64c3" />

## New Sample Data Prediction

<img width="656" height="343" alt="image" src="https://github.com/user-attachments/assets/120f5e0b-f6f2-4783-9964-77a9ad1a15a4" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
