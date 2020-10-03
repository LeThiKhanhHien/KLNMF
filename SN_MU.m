%%A hybrid Scalar Newton - Multiplicative Update algorithm for solving KL NMF problem
% SN-MU runs "options.SN_iterate" iterations of SN then take "options.MU_iterate" MU update.
% Input: X, r, options
% options include {display, init.W, init.H, maxiter, timemax, SN_iterate,MU_iterate,obj_compute}
% options.SN_iterate = 10 and options.MU_iterate =1 by default.
%
% Output: W, H (the factors), e (the objective sequence) and t (the sequence of
% running time)
% if options.obj_compute = 1 then the output (default value) then the output e would be 
% the sequence of the objective values;  otherwise, the output e would be [].
%
%
% written by LTK Hien
% Latest update: July 2020
function [W,H,e,t] = SN_MU(X,r,options) 
[m,n] = size(X); 

%% Parameters of NMF algorithm
if nargin < 3
    options = [];
end
if ~isfield(options,'display')
    options.display = 0; 
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
if ~isfield(options,'SN_iterate')
    options.SN_iterate =10; 
end
if ~isfield(options,'MU_iterate')
    options.MU_iterate =1; 
end
if ~isfield(options,'obj_compute')
    options.obj_compute = 1;  % if options.obj_compute = 0 then e = [], it means the objective is not evaluated. 
end
max_iter= options.SN_iterate; % default = 10 iterates of SN
max_time=50;
e=[];
t=[];
t_end=0;
obj_compute=options.obj_compute;
while t_end<options.timemax
    [W, H, e_SN ,t_SN]=SN(full(X),r,max_iter,max_time,W',H,5,0.2,obj_compute);
    W=W';
    e=[e e_SN];
    t=[t t_end + t_SN];

    % MU
    time1=tic; % time for MU update
    for j=1:options.MU_iterate
        rj=sum(W)';
        Wt=W';
        rjc=repmat(rj,1,n);
        bAx=X./(W*H+eps);
        cj=Wt*bAx; %r m m n
        H=max(eps,(H.*cj)./(rjc+eps));

        rj=(sum(H'))';
        rjc=repmat(rj,1,m); 
        bAx=X'./(H'*Wt+eps);
        cj=H*bAx;
        W=max(eps,((Wt.*cj)./(rjc+eps))');
    end
    t_end = t(end)+toc(time1); 
    t=[t t_end];
    if options.obj_compute==1
     e=[e KLobj(X,W,H)];
    end
end
end