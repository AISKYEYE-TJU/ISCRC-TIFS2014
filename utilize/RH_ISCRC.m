function label=RH_ISCRC(A,D,G,lambda1,lambda2,max_num)

%Regularized hull based ISCRC (RH-ISCRC-L1)
%This function is to solve:
% min{a,b}||Aa-Db||_2^2+lambda1||a||_lp+lambda1||b||_lp s.t. sum(a)=1
%Input: 
%A query image set
%D training image set dictionary 
%G sample number in each image set
%lambda1  regularization parameter
%lambda2  regularization parameter
%max_num  maximum iteration number

%Output:
%label prediction of A

beta2=5/size(A,2);
lam2=beta2/2;
A       =    A./ repmat(sqrt(sum(A.*A)),[size(A,1) 1]); % unit norm 2
D       =    D./ repmat(sqrt(sum(D.*D)),[size(D,1) 1]); % unit norm 2
for i=1:length(G)
index_l(i)=sum(G(1:i));
end
index_s=index_l;
index_s(2:end)=index_l(1:end-1)+1;
index_s(1)=1;

sample_mean=mean(D');
dis_As=slmetric_pw(A,sample_mean','sqdist');
[value,index]=min(dis_As);
a=zeros(size(A,2),1);
a(index)=1;
inter_num=0;

while inter_num<max_num 
inter_num=inter_num+1;
%The first step : fix a learn b 
y=A*a;

param.lambda = lambda1;
param.lambda2 = 0;
param.mode = 2;
param.L = size(A,1);
b = mexLasso(y, D, param);
% b = SolveDALM(D,y, 'lambda',lambda1,'tolerance',1e-3);

%The second step: fix b learn a 
y=D*b;
e=(beta2/2)^0.5*ones(1,size(A,2));
A_a=[A;e];
y_a=[y;(beta2/2)^0.5*(1-lam2/beta2)];

param.lambda = lambda2;
param.lambda2 = 0;
param.mode = 2;
param.L = size(A,1);
a = mexLasso(y_a, A_a, param);
% a = SolveDALM(A_a,y_a, 'lambda',lambda2,'tolerance',1e-3);

%the third step update lam
lam2=lam2+beta2*(sum(a)-1);
error=sum((A*a-D*b).^2);
end 

for j=1:length(G)
       error(j)=norm(A*a-D(:,index_s(j):index_l(j))*b(index_s(j):index_l(j))); 
end

[value,label]=min(error);