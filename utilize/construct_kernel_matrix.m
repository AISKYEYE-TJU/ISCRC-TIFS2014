function K=construct_kernel_matrix(X,Y,kernel_option)   
% This function constructs the kernel matrix by using selected
% kernel function type and its parameters.
%
% Inputs 
% X - dxM data matrix whose columns are samples in d-dimensional space
% Y - dxN data matrix whose columns are samples in d-dimensional space
% kernel_option - a struct type including kernel function parameters
%   kernel_option -  Choices are:
%        kernel_option.type = 'linear','polynomial','gaussian', and 'chi_square'
%        kernel_option.par = polynomial function degress such as '2','3' or
%        width of the exponential, q, i.e., exp(-DD/q).
% Outputs
% K - MxN kernel matrix
% 
% Written by Hakan cevikalp, 9/3/2007.
   
[d,N]=size(Y);
[d,M]=size(X);
if strcmpi(kernel_option.type,'linear'),
%     K=X'*Y;
    K=slmetric_pw(X, Y,'dotprod');
elseif strcmpi(kernel_option.type,'polynomial')
    K=X'*Y;
    K=K.^kernel_option.par;
elseif strcmpi(kernel_option.type,'gaussian')
    %K=EuclidDistance(X,Y);
    K=sqrDist(X,Y)';
%   K=slmetric_pw(X, Y,'sqdist');
    K=exp(-K/(2*kernel_option.par^2));
elseif strcmpi(kernel_option.type,'rbf')
    K=sqrDist(X,Y)';
    K=exp(-kernel_option.par*K);
elseif strcmpi(kernel_option.type,'chi_square');
%     for h=1:M,
%         K(h,:)=ChiSquare_Distance(X(:,h),Y);
%     end
    K=slmetric_pw(X, Y,'intersect');
%     K=exp(-K/kernel_option.par);
elseif strcmpi(kernel_option.type,'HIK');
    for i=1:size(X,1)
        for j=1:size(Y,1)
            K(i,j)=sum(min([X(:,i);Y(:,j)]))/size(X,2);
        end
    end
end