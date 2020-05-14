%% Create correct answer.
load answer

correct=zeros(208,1);
for i=1:208
    correct(i)=str2double(filesrand{i,2});
end

%% Create subject answer.
load subans

results=zeros(208,20);
for i=1:208
    for j=1:20
        if subans(i,j)~=correct(i)
            results(i,j)=subans(i,j);
        end
    end
end

%% Overall accuracy.

accuracy=results==0;
accuracy=sum(accuracy)/208;

% Overall accuracy not very high. Highest 68.75%. Mostly between 50% and
% 70%.

%% Accuracy by type.

type=zeros(20,8);
for i=1:20
    tmp=results(:,i);
    for j=1:8
        type(i,j)=sum(tmp(correct==j)==0)/sum(correct==j);
    end
end

typemean=mean(type);

%% Dataset validation

dv=zeros(8,8);
for i=1:8
    for j=1:8
        dv(i,j)=sum(sum(subans(correct==j,:)==i))/(20*sum(correct==j));
    end
end
% dv is the matrix that appears in the original CNN paper.

%% Corrcoef

corrparti=corrcoef(subans);
sortedcorr=sort(corrparti);
sortedcorr=sortedcorr(1:19,:);
ccmax=max(max(sortedcorr));
ccmin=min(min(sortedcorr));
ccmean=mean(mean(sortedcorr));

cccrct=zeros(1,20);
for i=1:20
    tmp=corrcoef(correct,subans(:,i));
    cccrct(i)=tmp(1,2);
end

%% Model Comparison - Data Preprocessing
vgg=importdata('VGG.xlsx');

actwise=vgg.Actor;
rndwise=vgg.Random;

rndname=rndwise(:,1);
rnd2324={};
tmp=1;
for i=2:length(rndname)
    tmpname=rndname{i};
    if (str2double(tmpname(19:20))==23)||(str2double(tmpname(19:20))==24)
        rnd2324(tmp,:)=rndwise(i,:);
        tmp=tmp+1;
    end
end

actname=actwise(:,1);
act2324={};
tmp=1;
for i=2:length(actname)
    tmpname=actname{i};
    if (str2double(tmpname(19:20))==23)||(str2double(tmpname(19:20))==24)
        act2324(tmp,:)=actwise(i,:);
        tmp=tmp+1;
    end
end

rnd2324=sortrows(rnd2324);
act2324=sortrows(act2324);

rnd=rnd2324;
act=act2324;
for i=1:length(rnd)
    if strcmp(rnd2324{i,2},'neutral')
        rnd{i,2}=1;
    elseif strcmp(rnd2324{i,2},'calm')
        rnd{i,2}=2;
    elseif strcmp(rnd2324{i,2},'happy')
        rnd{i,2}=3;
    elseif strcmp(rnd2324{i,2},'sad')
        rnd{i,2}=4;
    elseif strcmp(rnd2324{i,2},'angry')
        rnd{i,2}=5;
    elseif strcmp(rnd2324{i,2},'fearful')
        rnd{i,2}=6;
    elseif strcmp(rnd2324{i,2},'disgust')
        rnd{i,2}=7;
    else
        rnd{i,2}=8;
    end
    nm=rnd{i,1};
    nm(2)='2';
    rnd{i,1}=nm(1:20);
end
for i=1:length(act)
    if strcmp(act2324{i,2},'neutral')
        act{i,2}=1;
    elseif strcmp(act2324{i,2},'calm')
        act{i,2}=2;
    elseif strcmp(act2324{i,2},'happy')
        act{i,2}=3;
    elseif strcmp(act2324{i,2},'sad')
        act{i,2}=4;
    elseif strcmp(act2324{i,2},'angry')
        act{i,2}=5;
    elseif strcmp(act2324{i,2},'fearful')
        act{i,2}=6;
    elseif strcmp(act2324{i,2},'disgust')
        act{i,2}=7;
    else
        act{i,2}=8;
    end
    nm=act{i,1};
    nm(2)='2';
    act{i,1}=nm(1:20);
end

ansorder=filesrand(:,1);
for i=1:length(ansorder)
    tmp=ansorder{i};
    ansorder{i}=tmp(1:20);
end

ansact=zeros(208,1);
ansrnd=zeros(208,1);
ansactm=zeros(208,1);
ansrndm=zeros(208,1);

for i=1:208
    indact=strcmp(act(:,1),ansorder{i});
    ansact(i)=mode([act{indact,2}]);
    ansactm(i)=mean([act{indact,2}]);
end

for i=1:208
    indrnd=strcmp(rnd(:,1),ansorder{i});
    ansrnd(i)=mode([rnd{indrnd,2}]);
    ansrndm(i)=mean([rnd{indrnd,2}]);
end

% ansact(isnan(ansact))=0;
% ansrnd(isnan(ansrnd))=0;
% ansactm(isnan(ansactm))=0;
% ansrndm(isnan(ansrndm))=0;

ansrnd(isnan(ansrnd))=mean(ansrnd(isnan(ansrnd)==0));
% Replace the stimulus where there is no model response with group average.

%% Model Comparison with Human

accact=sum(ansact==correct)/208;
accrnd=sum(ansrnd==correct)/208;

ccactrnd=corrcoef(ansact,ansrnd);
ccactrnd=ccactrnd(1,2);

ccact=corrcoef(correct,ansact);
ccact=ccact(1,2);
ccrnd=corrcoef(correct,ansrnd);
ccrnd=ccrnd(1,2);

cchact=zeros(1,20);
for i=1:20
    tmp=corrcoef(ansact,subans(:,i));
    cchact(i)=tmp(1,2);
end
cchrnd=zeros(1,20);
for i=1:20
    tmp=corrcoef(ansrnd,subans(:,i));
    cchrnd(i)=tmp(1,2);
end

%% Response distribution

dvact=zeros(8,8);
for i=1:8
    for j=1:8
        dvact(i,j)=sum(sum(round(ansact(correct==j))==i))/(sum(correct==j));
    end
end

dvrnd=zeros(8,8);
for i=1:8
    for j=1:8
        dvrnd(i,j)=sum(sum(round(ansrnd(correct==j))==i))/(sum(correct==j));
    end
end