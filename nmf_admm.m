function [W, H, e, t] = nmf_admm(V, beta, rho,options, fixed)
% This code is taken from http://statweb.stanford.edu/~dlsun/admm.html
% Latest updated by LTK Hien, July 2020
%
%
% Implements NMF algorithm described in:
%   D.L. Sun and C. Févotte, "Alternating direction method of multipliers 
%      for non-negative matrix factorization with the beta divergence", ICASSP 2014.
%
% inputs
%    V: matrix to factor 
%    W, H: initializations for W and H
%    beta: parameter of beta divergence 
%          (only beta=0 (IS) and beta=1 (KL) are supported)
%    rho: ADMM smothing parameter
%    fixed: a vector containing the indices of the basis vectors in W to
%           hold fixed (e.g., when W is known a priori)
%
% outputs
%    W, H: factorization such that V \approx W*H
%  e (the objective sequence) and t (the sequence of running time) (Hien
%  added)
    cputime0=tic; % start the algorithm
    if ~isfield(options,'display')
        options.display = 0; 
    end
 
    W=options.init.W;
    H=options.init.H;
    e=[];
    timeerr=0;
    % determine dimensions
    [m,n] = size(V);
    [~,k] = size(W);
    t(1) = toc(cputime0);
    time1=tic;
    e(1)= KLobj(V,W,H);
    timeerr=toc(time1); % to remove the time of computing the objective 
  
    i=1;
    % set defaults
    if nargin<5, rho=1; end
    if nargin<6, fixed=[]; end
    
    % get the vector of indices to update
    free = setdiff(1:k, fixed);
    
    % initializations for other variables
    X = W*H;
    Wplus = W;
    Hplus = H;
    alphaX = zeros(size(X));
    alphaW = zeros(size(W));
    alphaH = zeros(size(H));

    while t(i) < options.timemax % run at most options.timemax seconds
        % update for H
        H = (W'*W + eye(k)) \ (W'*X + Hplus + 1/rho*(W'*alphaX - alphaH));
        
        % update for W
        P = H*H' + eye(k);
        Q = H*X' + Wplus' + 1/rho*(H*alphaX' - alphaW');
        W(:,free) = ( P(:,free) \ (Q - P(:,fixed)*W(:,fixed)') )';
        
        % update for X (this is the only step that depends on beta)
        X_ap = W*H;
        if beta==1
            b = rho*X_ap - alphaX - 1;
            X = (b + sqrt(b.^2 + 4*rho*V))/(2*rho);
        elseif beta==0
            A = alphaX/rho - X_ap;
            B = 1/(3*rho) - A.^2/9;
            C = - A.^3/27 + A/(6*rho) + V/(2*rho);
            D = B.^3 + C.^2;

            X(D>=0) = nthroot(C(D>=0)+sqrt(D(D>=0)),3) + ...
                nthroot(C(D>=0)-sqrt(D(D>=0)),3) - ...
                A(D>=0)/3;

            phi = acos(C(D<0) ./ ((-B(D<0)).^1.5));
            X(D<0) = 2*sqrt(-B(D<0)).*cos(phi/3) - A(D<0)/3;
        else
            error('The beta you specified is not currently supported.')
        end

        % update for H_+ and W_+
        Hplus = max(H + 1/rho*alphaH, 0);
        Wplus = max(W + 1/rho*alphaW, 0);
        
        % update for dual variables
        alphaX = alphaX + rho*(X - X_ap);
        alphaH = alphaH + rho*(H - Hplus);
        alphaW = alphaW + rho*(W - Wplus);
        
        i=i+1;
        
        time1=tic; 
        e(i)= KLobj(V,max(W,0),max(0,H));
        timeerr = timeerr+ toc(time1);
     
        t(i)= toc(cputime0)-timeerr;
        if  options.display ==1
         fprintf('ADMM: iteration %4d fitting error: %1.2e \n',i,e(i));  
        end
    end
    
    W(:,free) = Wplus(:,free);
    H = Hplus; 

end