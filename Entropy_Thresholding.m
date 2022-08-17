function [II, EE]=Entropy_Thresholding(Ws,ruo)
%%%%%From "A Density-Based Data Reduction Algorithm for Robust Estimators"
%%%%%Lecture Notes In Computer Science; Vol. 4478, 2007

%%%%%Input a set of weights of particles; 
%%%%%Output the threshold EE and the values used for the threshold II>EE.
% qq0=Ws.^2; 
qq0=Ws.^ruo; 
qq=max(qq0)-qq0; 
pp=qq/sum(qq); 
II=-log(pp+eps); 
EE=sum(pp.*II); 
