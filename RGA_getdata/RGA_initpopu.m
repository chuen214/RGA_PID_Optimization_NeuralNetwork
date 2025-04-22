function initpopu = RGA_initpopu(popu_size, gene_number, range) %產生初始族群
    initpopu = rand(popu_size, gene_number); %創建一個大小為 popu_size × gene_number 的隨機矩陣，矩陣中的值在0到1之間。
    for i = 1 : popu_size
        initpopu(i, :) = (range(2, :) - range(1, :)) .* initpopu(i, :) + range(1, :);
    end
end