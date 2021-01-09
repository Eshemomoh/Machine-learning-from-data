function [traindata,testdata] = constructdata(x1data1,x2data1,x1data0,x2data0)

testdata = [];
for i = 1:length(x1data1)
    testdata = [testdata;[x1data1(i),x2data1(i),1]];
end

for i = 1:length(x1data0)
    temp = [];
    temp = [temp;[x1data0(i),x2data0(i),-1]];
    testdata = [testdata;temp];
end
ndata = size(testdata,1);
index = randperm(ndata);
testdata = testdata(index,:);
index = randi(length(testdata),300,1);
index = fliplr(index);
traindata = testdata(index,:);
testdata(index,:) = [];

end