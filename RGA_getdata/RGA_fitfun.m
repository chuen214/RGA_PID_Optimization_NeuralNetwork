function [PI, result_data]  = RGA_fitfun(chro, target_value, result_data)
    global MIN_offset Kp Ki t y target_value result_data generation_no nn;
    
    MIN_offset = 9000;

    Kp = chro(1);
    Ki = chro(2);

    % 运行 Simulink 模型
    output = sim('PIDcontroller.slx');
    t = output.simout;   % 获取时间数据
    y = output.simout2;  % 获取系统输出（假设是位置、速度或其他控制变量）

    % 计算误差（目标值为 target_value）
    error = target_value - y;  % 使用传递的目标值进行误差计算

    % 计算误差积分（Integral of Error）
    integral_error = cumsum(error) * (t(2) - t(1));  % 使用 cumsum 计算误差积分

    % 计算过冲（overshoot）
    steady_state_value = y(end);  % 稳态值
    peak_value = max(y);          % 峰值
    overshoot = (peak_value - steady_state_value) / steady_state_value * 100;

    % 如果过冲过大，增加惩罚
    if overshoot > 10
        overshoot_penalty = 1000;
    else
        overshoot_penalty = 0;
    end

    % 适应度函数
    I = find(t > 3);  % 忽略前3秒的数据
    z = sum(abs(target_value - y(I))) + overshoot_penalty;  % 计算适应度

    % 将计算结果存储（target_value, Kp, Ki, error, integral_error 和 fitness score）
    if nn >= generation_no - 1
        if isempty(result_data) || ~any(result_data(:, 1) == Kp & result_data(:, 2) == Ki)
            new_data = [target_value, Kp, Ki, mean(error), mean(integral_error), z];
            % new_data = transpose(new_data);
            if isempty(result_data)
                result_data = new_data;
            else
                result_data = [result_data; new_data];  % 添加新数据
            end
        end
    end

    % 返回适应度值
    PI = MIN_offset - z;
end
