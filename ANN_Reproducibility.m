function A0a_ANN_OK % Solving the system, check

%% Boxplot for repeated analysis
%% Reproducibility of ANN outcome when bias and weights are not preinatilized

close all
clear,clc, format short g, format compact
profile on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tb = readtable('data.txt');
XY=tb{:,:};
X=XY(:,1:4);Y=XY(:,end);
%%
% For ANN data Each column is a sample, nu,ber of rows is number of feature
% similarly response in a row vector

[rank]=ksdesign(X,Y);



x=X'; t=Y';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
% Levenberg-Marquardt backpropagation.

%  x=x(:,:);t=t(:,:);
  
  


solu=[];

for j=1:500

%rng('default')
trainFcn = 'trainlm';
hiddenLayerSize = 6;
net0 = fitnet(hiddenLayerSize,trainFcn);
net0 = configure(net0,x,t);
net0.performFcn='mse';

% net0.layers{1}.transferFcn = 'tansig';
% net0.layers{2}.transferFcn = 'purelin';%%%'poslin';%'tansig';
% Setup Division of Data for Training, Validation, Testing

trn=sort(rank(1:20));val=sort(rank(21:25));test=sort(rank(26:30));
net0.divideFcn = 'divideind';
net0.divideParam.trainInd =trn;
net0.divideParam.valInd =val;
net0.divideParam.testInd=test;

% net0.divideParam.trainRatio = 66/100;
% net0.divideParam.valRatio = 17/100;
% net0.divideParam.testRatio = 17/100;

net0.trainParam.showWindow = 0; 
% Train the Network
[net1,tr] = train(net0,x,t);

% input_weight = net1.IW;
% layer_weight=net1.LW;
% biases = net1.b;

[ytr,etr,SSEtr,Rtr,R2tr,RMSEtr]=mynet_performance(net1,x(:,tr.trainInd),t(:,tr.trainInd)); %% only training data
[yva,eva,SSEva,Rva,R2va,RMSEva]=mynet_performance(net1,x(:,tr.valInd),t(:,tr.valInd)); %% only validation data
[y,e,SSE,R,R2tst,RMSEtt]=mynet_performance(net1,x(:,tr.testInd),t(:,tr.testInd)); %% entire data

solu=[solu;[RMSEtr,RMSEva, RMSEtt]];
end
boxplot(solu)
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(e)
% figure, plotregression(t,y)
%figure, plotfit(net1,x,t)
disp('');

function [y,e,SSE,R,R2,RMSE]=mynet_performance(net1,x,t)%% predictor and response
y = net1(x);
e = gsubtract(t,y);
SSE=sum(e.^2,2);
Yavg=mean(y,2);
SST=sum((y-Yavg).^2,2);
R2=(1-SSE/SST);
R=corrcoef(t,y);
MSE = perform(net1,t,y);%%Mean square error
RMSE=MSE^0.5;



function [rank]=ksdesign(X,Y)
XY=[X Y];
[~,b]=size(XY);
XY=sortrows(XY,b);
Xj=XY(:,1:end-1);
rank=ksrank(Xj);



function Rank=ksrank(X)
%+++ Employ the K-S algorithm for selecting the representative samples;
%+++ X: a m x n matrix with m samples and n variables.
%+++ Rank: sample index ordered by the representitiveness. if you want to select for example the most
%+++       representitive 10 samples, select the samples corresponding to
%+++       the first 10 indice in Rank.
%+++ Hongdong Li, lhdcsu@gmail.com, May 10,2008.

tic;
[Mx,~]=size(X);
Rank=zeros(1,Mx);
out=1:Mx;
D=distli(X);
[i, j]=find(D==max(max(D)));
Rank(1)=i(1);Rank(2)=j(1);
out([i(1) j(1)])=[];
%+++ Iteration of  K-S algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter=3;
while iter<=Mx
    in=Rank(Rank>0);
    Dsub=D(in,out);
    [minD,~]=min(Dsub);
    [~,indexmax]=max(minD);
    Vadd=out(indexmax);
    Rank(iter)=Vadd;
    out(out==Vadd)=[];
    iter=iter+1;
end
toc;
function D=distli(X)
X=X';
[~,N] = size(X);
X2 = sum(X.^2,1);
D = repmat(X2,N,1)+repmat(X2',1,N)-(2*(X'*X));
