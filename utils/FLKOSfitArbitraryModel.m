function [theta_f, sigma_f, Cinl, ht,  f_inx] = FLKOSfitArbitraryModel(x, k, model_type,Threshold,W, pinx)
% INPUTS
%x = data - D x N matrix of data; D - dimention of data ; N - Number of data points
%k - The size of the minimum acceptible structure in application
%model_type - type of model to fit
%'line2D', 'plane3D', 'homography', 'fundamental', 'subspace'
%Threshold = For dicotomizing inliers ffrom outliers:
%T = 2~3.0 assuming normally distribured noise;

% OUTPUTS
%theta_f - final set of paramers after one FLKOS iteration
%sigma_f - final sigma
%Cinl - indexes of the inlier to theta_f
%ht - the residuals with respect to theta_f
%f_inx - the sample that produced theta_f


[ fitfn, resfn, degenfn, psize, numpar ] = getModelParam(model_type);

[~,n] = size(x);
n_rand_inits = 1; %number of random initializations
n_iterations = 50; %number of flkos iterations

BestJ = 1e7;

old_k = k; %edit by ltt
dist_old = 0;

init_counter = 0;
while(init_counter < n_rand_inits)
    
    %initlialize from a random psize-tupple
    %[psub,pinx] = datasample(x,psize,2, 'Replace', false, 'Weights', W);
    %     pinx = randsample(n,psize);
    psub = x(:,pinx);
    
    iter_counter = 0;
    while( (iter_counter < n_iterations)  )
        theta = feval(fitfn,psub);  % Fit the model on the p-subset
        
        dist = feval(resfn,theta,x);% Compute residuals.
        dist_old = dist;
        
        %         throldValue = 0.001;
        %         endK = 30;
        %         if(sum(dist < throldValue)>k && k<endK)
        %             old_k = k; %edit by ltt
        %             k = sum(dist < throldValue);
        %             if(k > endK)
        %                 k = endK;
        %             end
        %         end
        %fprintf('currenct k = %d\t',k);
        [SRes,I] = sort(dist);
        pinx = I( (k-psize+1):k ); % Get new tupple around kth sorted residual
        psub = x(:,pinx);
        Jnew = sum( SRes( (k-psize+1):k ) ); % cost func - sum of p residuals(squared) around K
        
        %add by ltt
        %         endK = times*psize;
        %         if(k<endK)
        
        %         end
        
        
        if (Jnew < BestJ) % store the best value
            BestJ = Jnew;
            theta_f = theta;
            f_inx = pinx;
            
            s_dist = SRes;%sort(dist);
            if(sum(s_dist(1:psize))< 0.0005)  %0001
                [Cinl,sigma_f]  = getInliers(SRes,I,k,Threshold);
                Cinl = dist< (Threshold^2) * (sigma_f^2);
                conutInliers = sum(Cinl);
                if(conutInliers>k )%0.00005)%0.0001)%0.00005)
                    old_k = k; %edit by ltt
                    k = conutInliers;
                    %                 if(k > endK)
                    %                     k = endK;
                    %                 end
                end
            end
        end
        
        if (iter_counter ==0)
            old_pinx1 = pinx;
            old_pinx = pinx;
        end
        dist = feval(resfn,theta,x(:,old_pinx));
        dist = mean(dist);
        
        dist1 = feval(resfn,theta,x(:,old_pinx1));
        dist1 = mean(dist1);
        
        sd = SRes(k);%the kth residual with respect to current theta
        
        %stopping criterion
        %check if the tupples in last two interations are still within the
        %k residuals
        if ( ( dist < sd ) && ( dist1 < sd )  &&  (iter_counter > 10))
            break;
        end
        old_pinx1 = old_pinx;
        old_pinx = pinx;
        
        iter_counter = iter_counter+1;
    end
    %     if (Jnew < BestJ) % store the best value
    %             BestJ = Jnew;
    %             theta_f = theta;
    %             f_inx = pinx;
    %         end
    init_counter = init_counter+1;
end
%fprintf('currenct k = %d\t',k);
%get the best theta and refine it
ht = feval(resfn,theta_f,x);
[sresd2,sindx] = sort(ht);
[Cinl,sigma_f]  = getInliers(sresd2,sindx,k,Threshold);
% [ sigma_f, Cinl ] = Fast_AVG_MSSE( ht, Threshold, 100, 0, 2 );
%get the inliers based on refined theta
% psub = x(:,Cinl);
% theta_f = feval(fitfn,psub);
% ht = feval(resfn,theta_f,x);
% [sresd2,sindx]=sort(ht);
% [Cinl, sigma_f] = getInliers(sresd2,sindx,k,Threshold);
% psub = x(:,Cinl);
% theta_f = feval(fitfn,psub);
%disp(['k = ', num2str(k)])
%k2 = k
end
