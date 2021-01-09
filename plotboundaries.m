
%[trainset,testset] = getdata();
load onlinedata

j0 = find(trainset(:,3) ~= 1);
j1 = find(trainset(:,3) == 1);
x1 = linspace(-1.5,1,100); x2 = linspace(-1,1,100);
[X1,X2] = meshgrid(x1,x2);
XX1 = reshape(X1,[],1); XX2 = reshape(X2,[],1);
Z = zeros(length(X1));
for i = 1:length(X1)
    for j = 1:length(X1)
        Z(i,j) = frontproptest(model,[X1(i,j),X2(i,j)]');
    end
end

%Z = frontproptest(model,[XX1,XX2]');
%Z = reshape(Z,length(X1),[]);

close all
figure
plot(trainset(j1,1),trainset(j1,2),'bo')
hold on
plot(trainset(j0,1),trainset(j0,2),'rx')
contour(X1,X2,Z,1,'Color',	'#A2142F')

ylabel('Intensity')
xlabel('Symmetry')

legend('digit 1','other digit','Neural network')

for i = 1:length(testset)
    
yhat(i) = frontproptest(model,testset(i,1:2)');
end
ytrue = testset(:,end);
    missed = find(ytrue ~= yhat');
    Error = length(missed)/length(yhat);
    disp(Error)
    