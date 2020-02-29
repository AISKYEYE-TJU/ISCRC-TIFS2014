function label=KCH_ISCRC(A,B,kernel_option,C,trls,DD)

%kernerlized convex hull based ISCRC (KCH-ISCRC)
%This function is to solve:
% min{a,b}||fi(A)a-fi(B)b||_2^2 s.t. sum(a)=1 sum(b)=1 0<a<1 0<b<1 
%Input: 
%A query image set
%B training image set dictionary 
%kernel_option (default 'guassian')
%C  parameter for convex hull
%trls label of each sample in B
%DD  kernel matrix(D,D)

%Output:
%label prediction of A
%% unit norm 2 (A and B)
A       =    A./ repmat(sqrt(sum(A.*A)),[size(A,1) 1]);
B       =    B./ repmat(sqrt(sum(B.*B)),[size(B,1) 1]);
%% Gram matrix (Q)
[d,m]=size(A);
[d,n]=size(B);
Q=zeros(m+n,m+n);
Q(1:m,1:m)=construct_kernel_matrix(A,A,kernel_option);
temp=construct_kernel_matrix(A,B,kernel_option);
Q(1:m,m+1:m+n)=-temp;
Q(m+1:m+n,1:m)=Q(1:m,m+1:m+n)';
Q(m+1:m+n,m+1:m+n)=DD;
Aeq=zeros(2,m+n);
Aeq(1,1:m)=1;
Aeq(2,m+1:m+n)=1;
beq=[1 1]';
LB=0;
UB=C*ones(m+n,1);
%% QP problem (quadprog)
if  strcmpi(kernel_option.type,'linear')
opts = optimset('Algorithm','interior-point-convex','Display','off');
else strcmpi(kernel_option.type,'gaussian')
opts = optimset('Algorithm','interior-point-convex','Display','off');
end 
alpha=quadprog(2*Q,[],[],[],Aeq,beq,LB,UB,[],opts);
Q(1:m,m+1:m+n)=temp;
Q(m+1:m+n,1:m)=Q(1:m,m+1:m+n)';
a_t=ones(1,m);
%% classification
for k=1:max(trls)
b_t=zeros(1,n);
b_t(trls==k)=-1;
yy=[a_t b_t];
w=alpha.*yy';
distance(k)=sqrt(w'*Q*w);
end
[value,label]=min(distance);