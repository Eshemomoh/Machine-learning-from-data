function [wei1,wei2] = weightupdate(grad1,grad2,wei1,wei2,rate)

   wei1 = wei1 - rate*grad1;
   wei2 = wei2 - rate*grad2;
end
