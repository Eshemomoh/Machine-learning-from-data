function yhat = svmprediction(svmm,data,type,order)
nx = svmm.nsv;

%ny = size(x1,1); yhat = zeros(ny);
yhat = zeros(length(data),1);
for k = 1:length(data)
    What = 0;
        for i = 1:nx
            What = What + (svmm.alpha(i)'.*svmm.svy(i))*mykernel(svmm.svx(i,:),data(k,:),type,order);
        end
        yhat(k) = signx1(What + svmm.bias);


end