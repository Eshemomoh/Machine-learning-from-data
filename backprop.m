function [grad1,grad2] = backprop(lambda,model_activation,model_activation_dff,model_activation_fin,model_activation_dff_fin,...
wei1,wei2,grad1,grad2,datapoint,N)
% my backpropagation algorithm 
    y = datapoint(end);
 
    [hx,dff_sig3,dff_sig2,dff_sig1,sig1,sig2] = frontprop1(model_activation,model_activation_dff,model_activation_fin,model_activation_dff_fin,...
wei1,wei2,datapoint);
    
    err = 0.5*(hx - y)*dff_sig3;
    
    deltas3 = err;
  % for i = length(model.weights):-1:1
       
       smalldata2 = dff_sig2.*(wei2(2:end,:).*deltas3);
       deltas2 = smalldata2;
       
        
       smalldata1 = dff_sig1.*(wei1(2:end,:)*deltas2);
       deltas1 = smalldata1;
       
  % end


%for k = 1:length(model.layers)-1
    gnew1 = [1;sig1]*deltas2';
    grad1 = grad1+ gnew1/N + (lambda/(2*N))*wei1;
    
    gnew2 = [1;sig2]*deltas3';
    grad2 = grad2+ gnew2/N +(lambda/(2*N))*wei2;
    %Gradient = model.G;
%end














end