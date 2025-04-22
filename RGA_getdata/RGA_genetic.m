function [popu, fitness, upper, average, lower, BEST_popu] = RGA_genetic(popu_size, gene_number, range, fitfun, generation_no, crossover_rate, mutate_rate, elite)
    global nn
    initpopu = RGA_initpopu(popu_size, gene_number, range);
    popu = initpopu;
    
    upper = zeros(generation_no, 1);
    average = zeros(generation_no, 1);
    lower = zeros(generation_no, 1);
    BEST_popu = zeros(generation_no, gene_number);
    
    %計算在不同世代的族群適應度值
    for nn = 1 : generation_no
        [fitness, popu] = RGA_fitpopu(popu, fitfun);
        [upper(nn), index] = max(fitness);
        average(nn) = mean(fitness);
        lower(nn) = min(fitness);
        BEST_popu(nn, 1 : gene_number) = popu(index, :);
        popu = RGA_newpopu(popu, fitness, crossover_rate, mutate_rate, elite, range);
    end
end