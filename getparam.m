function [shift,scale] = getparam(data)
Min = min(data);
Max = max(data);
shift = (Max+Min)/2; %(Max-Min)/2 - Max;
scale = 2/(Max - Min);% 1/(Max+shift);


end