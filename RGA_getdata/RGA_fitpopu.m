function [fitness, popu] = RGA_fitpopu(popu, fitfun) %計算族群適應度值
    popu_size = size(popu, 1);
    fitness = zeros(popu_size, 1); 
    for I = 1 : popu_size
        fitness(I) = RGA_fiteach(popu(I, :), fitfun); %將每個染色體的適應度值存入 fitness 向量中
    end

    [fitness, index] = sort(fitness);
    popu = popu(index, :); %根據 index 將種群按適應度排序，由小到大。
end