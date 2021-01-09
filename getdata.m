function [trainset,testset] = getdata()

load digitdata
data = [traindata;testdata];
Ftrain = ComputeFeatures(data);
j0 = find(Ftrain(:,1)~= 1);
j1 = find(Ftrain(:,1)== 1);
intensity1 = Ftrain(j1,2);
intensity0 = Ftrain(j0,2);
symetry1 = Ftrain(j1,3);
symetry0 = Ftrain(j0,3);

% normalization
[shift,scale] = getparam([intensity1;intensity0]);
intensity1 = normalized(intensity1,shift,scale);
intensity0 = normalized(intensity0,shift,scale);
[shift,scale] = getparam([symetry1;symetry0]);
symetry1 = normalized(symetry1,shift,scale);
symetry0 = normalized(symetry0,shift,scale);

% construct data set
[trainset,testset] = constructdata(intensity1,symetry1,intensity0,symetry0);


end