%% Block Mirror Descent Method for solving KL NMF problem
% Input: X, r, options
% options include {display, init.W, init.H, maxiter, timemax,obj_compute}
% Output: W, H (the factors), e (the objective sequence) and t (the sequence of
% running time)
%
% if options.obj_compute = 1 then the output (default value) then the output e would be 
% the sequence of the objective values;  otherwise, the output e would be [].
%
% written by LTK Hien
% Latest update: July 2020
function [W,H,e,t] = BMD(X,r,options) 

cputime0 = tic; 
[m,n] = size(X); 

%% Parameters of NMF algorithm
if nargin < 3
    options = [];
end
if ~isfield(options,'display')
    options.display = 1; 
end
if ~isfield(options,'init')
    W = rand(m,r); 
    H = rand(r,n); 
else
    W = options.init.W; 
    H = options.init.H; 
end
if ~isfield(options,'maxiter')
    options.maxiter = 200; 
end
if ~isfield(options,'timemax')
    options.timemax = 5; 
end
if ~isfield(options,'obj_compute') % if it is 0 then BMD does not evaluate the objective, so e = []
    options.obj_compute = 1; 
end

i = 1; 


lambdaH=(1./(sum(X))); % the row which is the sum of columns of X
lambdaH=repmat(lambdaH,r,1);

lambdaW=(1./(sum(X,2)))';
lambdaW=repmat(lambdaW,r,1);

t(1) =toc(cputime0);
timeerr=0;
e=[];
if options.obj_compute==1
    time1=tic;
    e(1)= KLobj(X,W,H);
    timeerr=toc(time1); % to remove the time of computing the objective function
end
while i <= options.maxiter && t(i) < options.timemax  
   
    %update H
    H=updateH(X,W,H,lambdaH,n);
    %update W
    W=(updateH(X',H',W',lambdaW,m))';
    i=i+1; 
    if options.obj_compute==1
        time1=tic; 
        e(i)= KLobj(X,W,H);
        timeerr=timeerr + toc(time1);
    end
    t(i)= toc(cputime0)-timeerr;
    if  options.display ==1 && options.obj_compute==1
        if mod(i,100)==0
            fprintf('KL-nolips: iteration %4d fitting error: %1.2e \n',i,e(i));     
        end
    end

end
end
function H=updateH(X,W,H,lambdaH,n)
    rj=sum(W)';
    rjc=repmat(rj,1,n);
    bAx=X./(W*H+eps);
    cj=W'*bAx; 
    H=max(eps,H./(1+(lambdaH.*H).*(rjc-cj)));
end
