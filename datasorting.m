function[sorted,train_sort] = datasorting

%%
features = readcell('features.csv');
pnotes = readcell('patient_notes.csv');
train = readcell('train.csv');

titletrain = train(1,:);
titlepnotes = pnotes(1,:);
titlefeatures = features(1,:);

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
        if ~ismissing(pnotes{y(k2),1}) && ~ismissing(pnotes{y(k2),2}) && ~ismissing(pnotes(y(k2),3))
            sorted = [sorted;pnotes(y(k2),:)];
        end
    end
end

train_sort = {};
for k = 1:length(train)
    if ~strcmpi(train{k,6},'[]')
        train_sort = [train_sort;train(k,:)];
    end
end
%%
% for k = 1:length(train_sort)-1
%     %idx = find(cell2mat(features{:,1}== train_sort{k,4}))
%     train_sort{k,4} = features{cell2mat(features(:,1))== train_sort{k,4},3};
% end

%train_sort = [titletrain;train_sort];
writecell(sorted,'sorted_new.csv')