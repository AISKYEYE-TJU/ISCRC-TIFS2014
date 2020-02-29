function result=demo_mobo_KCHISCRC(num_frame,C,kerneltype,parncol)
%This function is to show how KCH-ISCRC works

%Input:
%num_frame  up to experimental setting (50 100 200)
%C          parameter for convex hull (defualt 1)  
%kerneltype (default 'gaussian') 
%parncol number of atoms in dictionary learning 

%Output:
%result classification accuracy

%usage
%result=demo_mobo_KCHISCRC(50,1,'gaussian',10) 
%result=demo_mobo_KCHISCRC(100,1,'gaussian',10) 
%result=demo_mobo_KCHISCRC(200,1,'gaussian',10) 
%% data preparation
addpath(genpath('database'))
addpath(genpath('utilize'))   

load MOBO
load rand_index
%% parameter setting
kernel_option.type=kerneltype; % selected kernel is gaussian
kernel_option.par=5;
%% random experiment (10 times)
for j=1:10
%get training set     
    ttls=[]; 
    tr_data=cell(1,25);
    D=[];
    labtr=[];
    G=ones(1,25);
    for i=1:25
       temp=MOBO{i,random_index{j}(i,1)};
       temp=temp(:,1:num_frame);
       if size(temp,2)>parncol
         tr_data{i}=dic_com(temp,parncol); 
       end
       D=[D,tr_data{i}];
       G(i)=size(tr_data{i},2);
       temp_l=ones(1,size(tr_data{i},2))*i;
       labtr=[labtr,temp_l];
       ttls((i-1)*3+1:i*3)=i;
    end
    ttls(75)=[];
%testing      
    D =D./ repmat(sqrt(sum(D.*D)),[size(D,1) 1]); % unit norm 2    
    DD=construct_kernel_matrix(D,D,kernel_option); 
    s=0;
    for i=1:24
        for k=2:4
          s=s+1;
          X1=MOBO{i,random_index{j}(i,k)};
          X1=X1(:,1:num_frame);
          lab=KCH_ISCRC(X1,D,kernel_option,C,labtr,DD);
          label(s)=lab;
        end
    end
    
    for i=25
        for k=2:3
          s=s+1;
          X1=MOBO{i,random_index{j}(i,k)};
          X1=X1(:,1:num_frame);
          lab=KCH_ISCRC(X1,D,kernel_option,C,labtr,DD);
          label(s)=lab;
        end
    end
    ClassRate(j)=sum(ttls==label)/length(label)
end

acuu_mean=mean(ClassRate);
acc_std=std(ClassRate);
result=[acuu_mean acc_std];