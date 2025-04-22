%% Step 1: 导入数据
data = readtable("D:\UserDATA\Desktop\New folder\optimization_results_rand.xlsx");  % 替换成你的文件路径

% 选择输入和目标数据
inputs = data{:, {'error', 'IntegralError'}};
targets = data{:, {'Kp', 'Ki'}};

% 数据归一化（将输入归一化到 [0, 1]）
inputs_scaled = normalize(inputs, 'range', [0 1]);

%% Step 2: 定义网络结构
layers = [
    featureInputLayer(2, 'Name', 'input')         % 输入层：2个特征
    fullyConnectedLayer(64, 'Name', 'fc1')          % 隐藏层：64个节点
    reluLayer('Name', 'relu1')                      % ReLU 激活函数
    dropoutLayer(0.3, 'Name', 'dropout')            % Dropout 层，防止过拟合
    fullyConnectedLayer(64, 'Name', 'fc2')          % 隐藏层：64个节点
    reluLayer('Name', 'relu2')                      % ReLU 激活函数
    fullyConnectedLayer(2, 'Name', 'fc3')           % 输出层：2个输出（Kp和Ki）
    regressionLayer('Name', 'regression')           % 回归层
];

%% Step 3: K折交叉验证设置
k = 5;  % 例如使用5折交叉验证
cv = cvpartition(size(inputs_scaled, 1), 'KFold', k);
valLosses = zeros(k,1);

for i = 1:k
    % 获取第 i 折的训练集和验证集索引
    trainIdx = training(cv, i);
    valIdx = test(cv, i);
    
    X_train = inputs_scaled(trainIdx, :);
    y_train = targets(trainIdx, :);
    X_val = inputs_scaled(valIdx, :);
    y_val = targets(valIdx, :);
    
    % Step 4: 设置训练选项，包含L2正则化和早停设置
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...                          % 最大训练周期数
        'InitialLearnRate', 0.001, ...
        'L2Regularization', 0.001, ...                  % L2正则化参数，可根据需要调整
        'ValidationData', {X_val, y_val}, ...           % 设置验证集
        'ValidationFrequency', 50, ...                  % 每50个迭代检查一次验证损失
        'ValidationPatience', 10, ...                   % 如果连续10次验证损失没有下降，则早停
        'Verbose', false, ...
        'Plots', 'none', ...                            % 不显示训练进度图（可以改为 'training-progress'）
        'Shuffle', 'every-epoch', ...
        'MiniBatchSize', 32, ...
        'ExecutionEnvironment', 'gpu');               % 使用 GPU 训练
    
    % Step 5: 训练模型
    net = trainNetwork(X_train, y_train, layers, options);
    
    % 评估验证集
    YPred = predict(net, X_val);
    valLosses(i) = mean((YPred - y_val).^2, 'all');
    
    fprintf('Fold %d: Validation Loss = %.4f\n', i, valLosses(i));
end

avgValLoss = mean(valLosses);
fprintf('Average Validation Loss: %.4f\n', avgValLoss);

%% （可选）使用全部训练数据训练最终模型
% 这里将全部数据用于训练，并利用交叉验证调参得到的设置
optionsFinal = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'InitialLearnRate', 0.0001, ...
    'L2Regularization', 0.001, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize', 32, ...
    'ExecutionEnvironment', 'gpu');
finalNet = trainNetwork(inputs_scaled, targets, layers, optionsFinal);

%% Step 7: 保存最终模型
save('pid_nn_model3.mat', 'finalNet');
disp('Training complete and model saved.');
