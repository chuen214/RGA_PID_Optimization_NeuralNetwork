import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_excel(r"D:\UserDATA\Desktop\學士專題\RGA_getdata\optimization_results_rand.xlsx", header=None)

# 將第一欄作為欄位名稱，後續資料轉置
data.columns = data.iloc[:, 0]  # 第一欄是欄位名稱
data = data.iloc[:, 1:]         # 移除第一欄
data = data.transpose()         # 把資料轉置為每列一筆樣本

# 欄位名稱自動轉為標準格式
data.columns.name = None
data.reset_index(drop=True, inplace=True)

inputs = data[['error', 'IntegralError']].values
targets = data[['Kp', 'Ki']].values

scaler = MinMaxScaler()
inputs_scaled = scaler.fit_transform(inputs)

X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class PIDNN(nn.Module):
    def __init__(self):
        super(PIDNN, self).__init__()
        self.fc1 = nn.Linear(2, 64) 
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

model = PIDNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
num_epochs = 3000
for epoch in range(num_epochs):
    model.train()
    
    outputs = model(X_train_tensor)
    
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

torch.save(model.state_dict(), 'pid_nn_model.pth')
print("Training complete and model saved.")
