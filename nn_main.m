% My neural network algorithm
% Author: Lucky Yerimah
% Date 05/12/2020

% load data
%cd 'C:\Users\lucky\Desktop\Matlab\Machine learning\SVM'
clear all
%
[trainset,testset] = getdata();
%%
%}
%trainset = [1 2 1]; for problem 1


% initialize the network
% network achitexture
m = 10; % single hidden layer with m nodes
N = size(trainset,1);
layers = [2,m,1]; 
activation = 'tanh';
activation_final = 'identity';
model = struct;
model = initialnet(model,N,layers,activation,activation_final);
%model initialization complete

% to reduce running time, we eliminate the struct and cell arrays

model_activation = model.activation; model_activation_dff = model.activation_dff;
model_activation_fin = model.activation_fin; model_activation_dff_fin = model.activation_dff_fin;
model_weights = model.weights; model_G = model.G; model_layers = model.layers; model_layers2 = model.layers2;
model_dff_layers = model.dff_layers; model_output = model.output;


wei1 = model_weights{1}; wei2 = model_weights{2}; grad1 = model_G{1}; grad2 = model_G{2};
sig1 = model_layers{1};sig2 = model_layers{2}; sig3 = model_layers{3};
sig21 = model_layers2{1};sig22 = model_layers2{2}; sig23 = model_layers2{3};
dff_sig1 = model_dff_layers{1}; dff_sig2 = model_dff_layers{2};dff_sig3 = model_dff_layers{3};


% do forward propagation and backpropagation and weight update

iteration = 2e6;
Ein = zeros(iteration,1);

[hx] = newfrontproptest(model_activation,model_activation_fin,wei1,wei2,trainset(:,1:2)'); % calculate first error
Error = errorf(hx,trainset(:,end)')/N;


Ein(1) = sum(Error);



counter = 0;
lambda = 0.01;
rate = 1; alpha = 1.1; beta = 0.8;
tic % stop watch timer
for k = 2:iteration
    counter = counter+1;
    disp(counter);
     
    for i = 1:N
        [grad1,grad2] = backprop(lambda,model_activation,model_activation_dff,model_activation_fin,model_activation_dff_fin,...
wei1,wei2,grad1,grad2,trainset(i,:),N);
    end
    
    
    % weight update
   % ;
    [wei1_temp,wei2_temp] = weightupdate(grad1,grad2,wei1,wei2,rate);
  
    [hx2] = newfrontproptest(model_activation,model_activation_fin,wei1_temp,wei2_temp,trainset(:,1:2)');
    Error2 = errorf(hx2,trainset(:,end)')/N;
    
   
    
    Newerror = sum(Error2);
   Ein(k) = Newerror + (lambda/(4*N))*(sum(wei1,'all') + sum(wei2,'all'));
    %Error = 0;
    
   if Ein(k) < Ein(k-1)
        wei1 = wei1_temp; wei2 = wei2_temp; rate = alpha*rate;
    else
        rate = beta*rate;
   end
   grad1 = grad1*0; grad2 = grad2*0;
    %}
    %model.G{1} = model.G{1}*0; model.G{2} = model.G{2}*0;
end
toc
model.weights{1} = wei1; model.weights{2} = wei2;

save('modelresults2b.mat','model','Ein')
%}

