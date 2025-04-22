function PI = RGA_fiteach(chro, fitfun)
    PI = feval(fitfun, chro); %PI = RGA_fiteach([1.2, 3.4, 2.8], 'myfitfun');
end