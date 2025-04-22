function new_popu = RGA_newpopu(popu, fitness, crossover_rate, mutate_rate, elite, range)
new_popu = popu;
popu_size = size(popu, 1);
gene_number = size(popu, 2);

%先找出最好的兩個染色體
if elite == 1
    tmp_fitness = fitness;
    [max1, index1] = max(tmp_fitness);
    tmp_fitness(index1) = min(tmp_fitness);
    [max2, index2] = max(tmp_fitness);
end

%天擇(輪盤法)
fitness_rate = fitness / sum(fitness);
fitness_rate_cum = cumsum(fitness_rate);

%交配(算術法)
for i = 1 : popu_size/2
    
    tmp = find(fitness_rate_cum > rand);
    parent1 = popu(tmp(1), :);
    tmp = find(fitness_rate_cum > rand);
    parent2 = popu(tmp(1), :);
    
    if rand < crossover_rate
        for J = 1 : gene_number
            new_popu(i*2-1, J) = parent1(J) + 0.1 * rand * (parent2(J) - parent1(J)); %1.5->0.1
            new_popu(i*2, J) = parent2(J) + 0.1 * rand * (parent1(J) - parent2(J)); %1.5->0.1
        end
    end
end

%突變(均勻法)
for i = 1 : popu_size
    for j = 1 : gene_number
        if rand <= mutate_rate
            new_popu(i, j) = range(1, j) + rand * (range(2, j) - range(1, j)) * 0.01;
        end
    end
end

if elite == 1
    new_popu([1 : 2], :) = popu([index1 index2], :);
end

end