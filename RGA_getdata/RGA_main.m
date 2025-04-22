clear, clc;

global MIN_offset Kp Ki t y u target_value result_data generation_no;

target_value = 94;
test_time = 10;
x = 1;

popu_size = 30;
gene_number = 2;
range = [-0.3 -0.05; 1.3 0.05]; %  第一行是各基因的下限，第二行是各基因的上限
fitfun = 'RGA_fitfun';
generation_no = 40;

for i = 1 : test_time
    result_data = [];
    crossover_rate = 0.8;
    mutate_rate = 0.02;
    elite = 1;

    disp(x);
    x = x + 1;

    [popu, fitness, upper, average, lower, BEST_popu] = RGA_genetic(popu_size, gene_number, range, fitfun, generation_no, crossover_rate, mutate_rate, elite);
    
    minfitness = MIN_offset - upper;
    [minimum_f, generation] = min(minfitness);
    minimum_Kp_Ki_Kd = BEST_popu(generation, :);
    
    Kp = minimum_Kp_Ki_Kd(1);
    Ki = minimum_Kp_Ki_Kd(2);
    
    if ~isempty(result_data)
        result_table = array2table(result_data, 'VariableNames', {'target_value', 'Kp', 'Ki', 'error', 'IntegralError', 'FitnessScore'});
        
        % 如果文件已经存在，使用 'append' 模式
        if exist('optimization_results_rand.xlsx', 'file') == 2
            writetable(result_table, 'optimization_results_rand.xlsx', 'WriteMode', 'append');
        else
            % 如果文件不存在，则创建并写入数据
            writetable(result_table, 'optimization_results_rand.xlsx');
        end
    end
    target_value = target_value + (1 + 2*rand);
end

figure(1)
tt = 1 : generation_no;
plot(tt, minfitness, '*-');
title('Minmum of PI = sum(abs(100 * (1 - y(I))))');
ylabel('PI');
xlabel('Generation');
    
output = sim('PIDcontroller.slx');
t = output.simout;
u = output.simout1;
y = output.simout2;
    
figure(2)
plot(t, y);
title('step response');
xlabel('time');
ylabel('y');
    
figure(3)
plot(t, u);
title('control energy');
xlabel('time');
ylabel('u');