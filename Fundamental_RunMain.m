
close all;
% addpath('./model_specific');
% addpath('./data');
% addpath('./LPM');
% addpath('./utils');

currentFolder = pwd;
addpath(genpath(currentFolder));
run('vl_setup');
rng(22)

load('fundLabel2');

results = zeros(19,5);
for seq_num = 1:19 %2 8
    disp(['running seq: ', cell2mat(fundLabel2(seq_num))])
    load (cell2mat(fundLabel2(seq_num))); %toycubecar.mat;  boardgame.mat;% biscuitbookbox.mat;%
    [data,ia,ic] = unique(data','rows');
    [LPMCorrectIndex] = LPM(data');
    numModels=max(label) - min(label);
    
    numPoints=[];
    for i=min(label):max(label)
        numPoints = [ numPoints , sum(label==i)];
    end
    disp(['Num Points(outliersFirst): ', num2str(numPoints)])
    N = sum(numPoints);
    
    %Parameter Declaration
    k = floor(min(0.1*N, 10));


    model_type = 'fundamental';
    Threshold = 2.5;
    SampFrac_min = 1/numModels;
    numHypo = 100; %number of hypothesis to be generated for clustering
    
    
    %remove repeating rows in data
    data = data';
    label = label(ia);
    
    %data1 = data(:,LPMCorrectIndex);%add by ltt
    
    dat_img_1 = normalise2dpts(data(1:3,:));
    dat_img_2 = normalise2dpts(data(4:6,:));
    
    X = [dat_img_1(:,LPMCorrectIndex);dat_img_2(:,LPMCorrectIndex)];
    X_ALL = [dat_img_1;dat_img_2];
    
    numRun = 100;
    miss_rateH = zeros(1,numRun);
    ttimeH = zeros(1,numRun);
    for nRun=1:numRun        
        [ClustLabels,ttime] = Fundamental_Run(LPMCorrectIndex, k, numModels, model_type, Threshold, SampFrac_min, numHypo, X_ALL);
        ClustLabels = ClustLabels-1;
        
        %Permute data labels to match the originals.
        %[miss_rate,index] = missclass(ClustLabels,label);
        miss_rate = Misclassification(ClustLabels+1,label+1);
%         new_elabel = zeros(size(ClustLabels));
%         for i=1:max(ClustLabels)
%             new_elabel(ClustLabels == index(i+1)) = i;
%         end
%         ClustLabels = new_elabel;
        
        miss_rateH(nRun) = miss_rate;
        ttimeH(nRun) = ttime;
        disp(['misclass error = ', num2str((100*miss_rate))])
    end
    
    sFileName = strcat(cell2mat(fundLabel2(seq_num)), '_results.mat');
    %save(sFileName, 'miss_rateH' , 'ttimeH')
    results(seq_num,:) = [median(miss_rateH), mean(miss_rateH), std(miss_rateH), median(ttimeH), mean(ttimeH)];
    %save('results.mat','results');
    disp(['mean misclass error = ', num2str(mean(miss_rateH))])
    disp(['median misclass error = ', num2str(median(miss_rateH))])
end
disp(['mean misclass error = ', num2str(mean(results(:,2)))])
disp(['median misclass error = ', num2str(median(results(:,1)))])

figure
subplot(2,2,1);
% img1=imresize(img1,1/6);
imshow(img1);hold on
gscatter(data(1,:), data(2,:), label,[],[],20)
title('ground truth')

subplot(2,2,2);
%figure
imshow(img1);hold on
gscatter(data(1,:), data(2,:), ClustLabels,[],[],20)
title('Estimated Clusters')

