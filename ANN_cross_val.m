function A2_ANN_cross_val_OK % Solving the system, check
close all hidden
clear,clc, format short g, format compact
global x t
profile on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ANN modelling for cross validation - full model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tb = readtable('data.txt');
XY=tb{:,:};
X=XY(:,1:4);Y=XY(:,end);
%%
% For ANN data Each column is a sample, nu,ber of rows is number of feature
% similarly response in a row vector
[rank]=ksdesign(X,Y);
x=X';t=Y';

cross(rank);%% cross validation module

disp('');

function cross(rank)


global count t val
count=1;
tt=rank(1:25)';
val=rank(26:30);

mcrep=20; %montecarlo repear
f=@(Xtrain,Xtest)cvsqrerr(Xtrain,Xtest);
PCRmsep=crossval(f,tt,'kfold',5,'mcreps',mcrep);%%,'mcreps',10
SSE=sum(PCRmsep,1)/mcrep;


Rdata=t(:,tt);
SSA=sum((Rdata-mean(Rdata)).^2,2);

RMSE=(SSE/size(tt,1)).^0.5;
R2CV=1-(SSE/SSA);disp('');
fprintf('The RMSE %2.4f and R2CV is %2.4f\n',RMSE,R2CV);

function SSEtr = cvsqrerr(trn,test)
global count x t val


rng("default")
trainFcn = 'trainlm';
hiddenLayerSize = [6];
net0 = fitnet(hiddenLayerSize,trainFcn);
net0 = configure(net0,x,t);
net0.performFcn='mse';


net0.divideFcn = 'divideind';
net0.divideParam.trainInd =trn;
net0.divideParam.valInd =val;
net0.divideParam.testInd=test;
net0.trainParam.showWindow = 0; 
% Train the Network
[net1,~] = train(net0,x,t);

[~,~,SSEtr,~,~]=mynet_performance(net1,x(:,test),t(:,test)); %%only training data

fprintf('The %d, fold with %d samples  \n',count,size(x(:,test),2));
count=count+1;

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
% crank=sort(rank(1:m)); %% was 1:35
% vrank=sort(rank(m+1:end));%% was 36:end

% Xco=Xj(crank,:);Xvo=Xj(vrank,:);
% Yco=Yj(crank);Yvo=Yj(vrank);
disp('');


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
