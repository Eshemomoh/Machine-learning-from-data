
       function model = initialnet(model,N,layers_nod_count,activation,activation_fin)
            act = @(x) tanh(x);
            dff_act = @(x) 1 - (tanh(x)).^2; % derivative of tanh(x)
            logistic = @(x) 1./(1 + exp(-x));
            dff_logistic = @(x) logistic(x).*(1 - logistic(x));
            identity = @(x) x;
            dff_identity = @(x) ones(size(x));
            signx = @(x) signx1(x); 
            
           % middle layer activation
           if activation == "logistic"
               model.activation = logistic;
               model.activation_dff = dff_logistic;
           elseif activation == "tanh"
               model.activation = act;
               model.activation_dff = dff_act;
           elseif activation == "identity"
                model.activation = identity;
               model.activation_dff = dff_identity;
           end
               
           
           % final layer activation
           if activation_fin == "logistic"
               model.activation_fin = logistic;
               model.activation_dff_fin = dff_logistic;
           elseif activation_fin == "tanh"
               model.activation_fin = act;
               model.activation_dff_fin = dff_act;
           elseif activation_fin == "identity"
               model.activation_fin = identity;
               model.activation_dff_fin = dff_identity;
           elseif activation_fin == "sign"
               model.activation_fin = signx;
               model.activation_dff_fin = 0;
           end
           
           % create empty array to hold weights and biases
           
           model.weights = {1,layers_nod_count-1};
           model.G = {1,layers_nod_count-1};

           for i = 1:length(layers_nod_count)-1
              w =  0.25*ones(layers_nod_count(i)+1,layers_nod_count(i+1));
              %w = rand(layers_nod_count(i)+1,layers_nod_count(i+1));
              model.weights{i} = w;
              model.G{i} = 0*w;
           end
           
           model.layers = cell(1,length(layers_nod_count));
           model.layers2 = cell(1,length(layers_nod_count));
           % store values of every node
           for i = 1:length(layers_nod_count)
               layer = zeros(layers_nod_count(i),1);
               model.layers{i} = layer;
               model.layers2{i} = zeros(layers_nod_count(i),N);
           end
           
           model.dff_layers = cell(1,length(layers_nod_count));
           
           % store values of every node
           for i = 1:length(layers_nod_count)
               dff_layer = zeros(layers_nod_count(i),1);
               model.dff_layers{i} = dff_layer;
           end
           
           model.output = signx(model.layers{end});
           
           
       end
       
      
