function  result=demo_mobo_RHISCRC(num_frame,lambda1,lambda2,parncol)
%This function is to show how KCH-ISCRC works

%Input:
%num_frame  up to experimental setting (50 100 200)
%lambda1 regularization parameter           
%lambda2 regularization parameter         
%parncol number of atoms in dictionary learning 

%Output:
%result classification accuracy

%usage
%result=demo_mobo_RHISCRC(50,0.001,0.001,10) 
%result=demo_mobo_RHISCRC(100,0.001,0.001,10) 
%result=demo_mobo_RHISCRC(200,0.001,0.001,10) 
%% data preparation
addpath(genpath('database'))
addpath(genpath('utilize'))   
load MOBO
load rand_index
%% parameter stting
max_num=20;
%% random experiment (10 times) 
ClassRate=zeros(1,10);
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
    
% testing 
    D= D./ repmat(sqrt(sum(D.*D)),[size(D,1) 1]); % unit norm 2
    s=0;
    for i=1:24
        for k=2:4
          A=MOBO{i,random_index{j}(i,k)};
          if size(A,2)>num_frame
           A=A(:,1:num_frame);
          end
          s=s+1;
          label(s)=RH_ISCRC(A,D,G,lambda1,lambda2,max_num);
        end
    end
    
     for i=25
        for k=2:3
          A=MOBO{i,random_index{j}(i,k)};
          if size(A,2)>num_frame
           A=A(:,1:num_frame);
          end
          s=s+1;
          label(s)=RH_ISCRC(A,D,G,lambda1,lambda2,max_num);
        end
    end
    
    ClassRate(j)=sum(ttls==label)/length(label)   
end

acuu_mean=mean(ClassRate);
acc_std=std(ClassRate);
result=[acuu_mean acc_std];