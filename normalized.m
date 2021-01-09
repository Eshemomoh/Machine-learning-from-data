function D = normalized(data,shift,scale)
for i = 1:length(data)
    D(i,:) = scale*(data(i)- shift);
end
end