
function svmmod = svmclassifier(x,y,C,type,order)
%%%%%%%%%%%%%%%%%%%%%%%
% Author: Lucky Yerimah
% my support vector algorithm
% ----------------------------
% Input:
% x: features from training data
% y: output of each feature set
% C: regularization parameter
% order: order of polynomial kernel
% svmmod is struct of svm model


%%x = [ones(size(x,1),1) x];
[nx] = size(x,1);
% setup optimization paramenters
Q = getQmat(x,y,type,order);
f = -ones(nx,1);
LB = zeros(nx,1);
UB = C*ones(nx,1);
A = -[y';-y';eye(nx)];
b = zeros(nx+2,1);

Aeq = y';
beq = 0;
x0 = zeros(nx,1);
options = optimoptions(@quadprog,'OptimalityTolerance',1e-14,'Display','off');

alpha = quadprog(Q,f,A,b,Aeq,beq,LB,UB,x0,options);

% calculate weights
epsilon = 1e-6;
index = find(alpha>epsilon);
svmmod.alpha = alpha(index);
svmmod.svx = x(index,:);
svmmod.svy = y(index);
svmmod.nsv = length(index);

 bias = zeros(1,length(index));
for j = 1:length(index)
    mu = 0; 
    for i = 1:length(index)
        mu = mu + (svmmod.alpha(i)'.*svmmod.svy(i))*mykernel(svmmod.svx(i,:),...
            svmmod.svx(j,:),type,order);
    end
    bias(j) = svmmod.svy(j) - mu;
end
svmmod.bias = mean(bias);








end