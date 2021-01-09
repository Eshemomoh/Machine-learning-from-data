% polynomial kernel

function kx = mykernel(x1,x2,type,order)

switch type
    case 'linear'
        kx = x1*x2';
    case 'poly'
        kx = (1 +dot(x1,x2))^order;
    case 'rbf'
        kx = exp(-order*(norm(x1-x2))^2);
end

end