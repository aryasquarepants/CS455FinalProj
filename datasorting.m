clc;
clear;

features = readcell('features.csv');
pnotes = readcell('patient_notes.csv');
train = readcell('train.csv');

train(1,:) = [];
pnotes(1,:) = [];
features(1,:) = [];



x = train(:,3);
c = unique(cell2mat(x));
pnnum_pnotes = cell2mat(pnotes(:,1));

sorted = {};

for k = 1:length(c)
     y = find(pnnum_pnotes == c(k));
     for k2 = 1:length(y)
         sorted = [sorted;pnotes(y(k2),:)];
     end
end
     
writecell(sorted,'sorted.csv')