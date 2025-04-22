import torch
import torch.onnx
import torch.nn as nn

class PIDNN(nn.Module):
    def __init__(self):
        super(PIDNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 输入层：2个特征（error, IntegralError）
        self.fc2 = nn.Linear(64, 64) # 隐藏层
        self.fc3 = nn.Linear(64, 2)  # 输出层：2个神经元（Kp, Ki）
        self.dropout = nn.Dropout(0.3)  # Dropout 层，防止过拟合

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = self.dropout(x)  # Dropout
        x = torch.relu(self.fc2(x))  # ReLU 激活函数
        x = self.fc3(x)  # 输出 Kp 和 Ki
        return x

# 加载训练好的模型
model = PIDNN()
model.load_state_dict(torch.load(r"D:\UserDATA\Desktop\學士專題\Neural_Network\PIDNN.py"))
model.eval()

# 输入数据示例
dummy_input = torch.randn(1, 2)  # 假设输入为 (error, integral_error)

# 将模型保存为 ONNX 格式
torch.onnx.export(model, dummy_input, 'pid_nn_model.onnx')
