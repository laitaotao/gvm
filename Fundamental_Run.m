function [ClustLabels,ttime] = Fundamental_Run(LPMCorrectIndex, k, numModels, model_type, Threshold, SampFrac_min, numHypo, X_ALL)

%INPUTS
%X - Input Data [d,N]
%k - paprameter kth
%numModels - number of models to recover
%model_type - model type - line2D, fundamental, homography ....
%Threshold - MSSE theshold T
%SampFrac_min - sample fraction to be included in bootstrap
%numHypo - number of hypothesis to be generated

%OUTPUTS
%ClustLabels - estimated labels (not corrected with original
%ttime - time to run the algorithm


SampFrac = SampFrac_min; %fraction of data to be included in a sample
USE_GRAOUSE =0;

[ fitfn, resfn, degenfn, psize, numpar ] = getModelParam(model_type);

X = X_ALL(:,LPMCorrectIndex);
N = size(X, 2);
W = ones(1,N); %weights for bootstrap sampling

H = zeros(size(X_ALL, 2),numHypo); %container for holding affinities to models
sdevHold = zeros(1,numHypo); %container for holding std to models

sigmaExp = 0.5;
[nearPtsTab] = calcNearPtsTab(X(1:2,:), 'exp', sigmaExp);
hypo_count=1;
tic;
Converged = 0;
remainingPoints = 1:length(X(1,:));
while (Converged==0)
    %sample a dataset according to weights
    %     [Xs,Is] = datasample(X, floor(SampFrac*N), 2, 'Replace', false, 'Weights', W);
    %     Ws = W(Is);
    if (length(remainingPoints)<= k)
        remainingPoints = 1:length(X(1,:));
        W = ones(1,N);
    end
    Xout = X(:,remainingPoints);
    Ws = W(remainingPoints);
    nearPtsT = nearPtsTab(remainingPoints,remainingPoints);
    choice = randperm(length(remainingPoints),1);
    nearPointsCdf = nearPtsT(choice,:);% + nearPtsTab2(ind,:))/2;
    nearPointsCdf(choice) = 0;
    nearPointsCdf = nearPointsCdf / sum(nearPointsCdf);
    nearPointsCdf = cumsum(nearPointsCdf);
    rndSub = rand(psize,1);
    [dum, pinxSub] = histc(rndSub,[0 nearPointsCdf]);
 
    %run HMSS on selcted data sample
    [theta_f, sigma_f, ins , ~,  ~] = FLKOSfitArbitraryModel(Xout, k, model_type,Threshold, Ws,  pinxSub);%2);
    identPoints = remainingPoints(ins);    
    remainingPoints = setdiff(remainingPoints ,identPoints);

    %do MCMC rejection
    %{
    u = rand(1);
    p = min(1, exp( -(sigma_f-sigma_o)) );
    sigma_o = sigma_f;
    if(u>p && hypo_count ~= 0)
       continue;
    end
    %}
    
    %get the residual for all data points
    %ht=feval(resfn, theta_f, X);
    ht_ALL=feval(resfn, theta_f, X_ALL);
    ht = ht_ALL(LPMCorrectIndex);
    
    %calculate the inliers and outlier (all data points are used here)
    Cinl = ht< (Threshold^2) * (sigma_f^2);
    conutInliers = sum(Cinl);
    Coutl = ~Cinl;
    
    %******************************use Eq. (11) ***************************************%
    H(:,hypo_count) = exp(-ht_ALL/( 2*(sigma_f^2) ) );
    sdevHold(hypo_count) = sigma_f;
    
    %update the bootstraping weights
    W(Cinl) = W(Cinl)/2;
    W(Coutl) = W(Coutl)*2;
    W(W<.1) = .1;
    W(W>20) = 1;
    
    
    if hypo_count==numHypo
        Converged = 1;
    end
    
    
    hypo_count = hypo_count+1;
end

%******************************construct affinity matrix G by H***************************************%
G = H*H';

%******************************construct affinity matrix G by H***************************************%
for sk = 1: size(G,1)
    vec = G(sk,:);
    vec(sk) = 0;
    [II4, EE4]=Entropy_Thresholding(vec, 2);
    sig_index4 = find(II4>EE4);
    G(sk,:)= 0;
    G(sk,sig_index4)= vec(sig_index4);%/sum(vec(sig_index));
    G(sk,sk) = 0;
end
CKSym = (G + G');%/2;

[~, ClustLabels] = spectralClustering_ALI(CKSym, numModels);

ttime = toc;







