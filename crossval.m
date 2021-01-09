function Ecv = crossval(data,lambda)
    counter = 0;
    n = size(data,1);
    
    for i = 1:n
       temp = data;
       temp(i,:) = [];
       g = lireg(temp,lambda);
       if accuracy(g,data(i,:)) == 1
           counter = counter +1;
       end
    end
    naccurate = counter/length(data); Ecv = 1-naccurate;


end