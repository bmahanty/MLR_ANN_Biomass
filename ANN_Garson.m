function A1c_ANN_OK % Solving the system, check
close all
clear,clc, format short g, format compact
profile on
rng('default')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ANN variable of importance Garson Eq.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tb = readtable('data.txt');
XY=tb{:,:};
X=XY(:,1:4);Y=XY(:,end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[rank]=ksdesign(X,Y);



x=X'; t=Y';
% Create a Fitting Network
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


net0.trainParam.showWindow = 0; 
% Train the Network
[net1,~] = train(net0,x,t);

IW = abs((net1.IW{1,1})); %nuron (j) * predictor (i)
LW = abs((net1.LW{2,1})); %% for each predictor row vector
biases = net1.b;


I=[];

for i=1:4
Ii=sum((IW(:,i)./sum(IW,2)).*LW','all');
I=[I Ii];
end

I=I/sum(I,'all');
fprintf('relative importance of variables: %s\n', sprintf('%4.3f ', I))
disp('');

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
