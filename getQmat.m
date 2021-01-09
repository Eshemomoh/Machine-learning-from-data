% generate Q matrix for optimization

function Qmat = getQmat(x,y,type,order)


[nx] = size(x,1);

Qmat = [];
for i = 1:nx
    qlit = zeros(nx,1);
    for j = i:nx
        qlit(j) = y(j)*y(i)*mykernel(x(j,:),x(i,:),type,order);
    end
    Qmat = [Qmat qlit];

    
        
end

end