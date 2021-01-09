 %creat a function that returns finaly layer activation values such
       %that the data dimension should be consistent with the first input
       %layer dimension
       function [answer] = newfrontproptest(model_activation,model_activation_fin,wei1,wei2,datapoint)
       
          x = datapoint; %y = datapoint(:,end);
        
             sig1 = x;
             %dff_sig1 = model_activation(sig1);  
          
          % propagation
          %for i = 1:length(model.weights)
             % w1 = model.weights{1};
             nx = size(x,2);
              input_layer1 = [ones(1,nx);sig1];
              %b = model.biases{i};
              next_layer1 = (wei1'*input_layer1);% + b;
              %dff_next_layer1 = model_activation_dff(next_layer1);
              next_layer1 = model_activation(next_layer1);
              sig2 =  next_layer1; 
              %dff_sig2 = dff_next_layer1;
              
              
              %w2 = model.weights{2};
              input_layer2 = [ones(1,nx);sig2];
              next_layer2 = (wei2'*input_layer2);
              %dff_next_layer2 = model_activation_dff_fin(next_layer2);
              next_layer2 = model_activation_fin(next_layer2);
              sig3 =  next_layer2;
              %dff_sig3 = dff_next_layer2;
              answer = sig3;
              %{
               
              if i == length(model.weights)
                  dff_next_layer = model.activation_dff_fin(next_layer);
                  next_layer = model.activation_fin(next_layer);
              else 
                  dff_next_layer = model.activation_dff(next_layer);
                  next_layer = model.activation(next_layer);
              end
              model.layers{i+1} = next_layer;
              model.dff_layers{i+1} = dff_next_layer;
          end
              %}
             
          %answer = model.layers{end};
              
       end