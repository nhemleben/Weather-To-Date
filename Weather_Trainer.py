import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load your dataset
# Your CSV should have columns like: temp_max, temp_min, humidity, wind_speed, day_of_year
df = pd.read_csv("weather_data.csv")

# Features and target
X = df[["temp_max", "temp_min", "humidity", "wind_speed"]].values
y = df["day_of_year"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert target to tensor (regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the model
class WeatherNet(nn.Module):
    def __init__(self):
        super(WeatherNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Regression output
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer
model = WeatherNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            val_loss = criterion(model(X_test_tensor), y_test_tensor)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()
    actuals = y_test_tensor.numpy().flatten()
    mae = np.mean(np.abs(predictions - actuals))
    print(f"Mean Absolute Error on Test Set: {mae:.2f} days")
