function [grdw] = grdfrontprop_p1a(model,datapoint,rate)

% x = datapoint(:,1:end-1); 
y = datapoint(end);
 
 [hx,~] = frontprop(model,datapoint);
 err = errorf(hx,y);
 
 grdw = cell(1,length(model.weights));
 
 for i = length(model.weights):-1:1
     grdwmat = zeros(size(model.weights{i}));
     
     for j = 1:size(model.weights{i},2)
         
         grdwmat_row = zeros(length(model.weights{i}(:,j)),1);
         
         for k = 1:length(model.weights{i}(:,j))
            model.weights{i}(k,j) = model.weights{i}(k,j) + rate;
            [hx_nod,~] = frontprop(model,datapoint);
            err_nod = errorf(hx_nod,y);
            model.weights{i}(k,j) = model.weights{i}(k,j) - rate;
            
            grdwmat_row(k) = (err_nod-err)/rate;
             
         end
        grdwmat(:,j) = grdwmat_row;
         
     end
     grdw{i} = grdwmat;
     disp(grdw)
     disp(model.weights)
 end
 
     %{
     bias_grd = zeros(size(model.biases));
     
     for i = 1:size(model.biases,2)
        bias_vec = zeros(length(model.biases(:,i)),1);
        
        for j = 1:length(model.biases(:,i))
            
            model.biases(j,i) = model.biases(j,i) - rate;
            hx_node = frontprop(model,datapoint);
            err_node = errorf(hx_node,y);
            model.biases(j,i) = model.biases(j,i) - rate;
            bias_vec(j) = (err_node - err)/rate;
                       
        end
        bias_grd(:,i) = bias_vec; 
        
     end
     
     disp(bias_grd)
     disp(model.biases)
     
     %}
     

end