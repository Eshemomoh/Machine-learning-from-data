function Error = errorf(hx,y)
           %yhat = hx(1,:);
           Error = 0.25*(hx - y).^2;
       end