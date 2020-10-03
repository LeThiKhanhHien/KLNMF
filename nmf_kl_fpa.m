function [W, H, obj, time] = nmf_kl_fpa(V, W, H, N, D,options)
% This code is taken from https://github.com/felipeyanez/nmf
% Latest updated by LTK Hien, July 2020
% [W, H, obj, time] = nmf_kl_fpa(V, W, H, N, D)
%
%% Non-negative matrix factorization (NMF) implementation using a 
% first-order primal-dual algorithm (FPA).
%
% Given a non-negative matrix V, find non-negative matrix factors W and H 
% such that V approx. W*H, i.e. solving the following optimization problem:
%
% min_{W,H} D(V||W*H),
%
% where D(V||W*H) is the Kullback-Leibler divergence loss.
%
% The FPA estimates W and H at each iteration, solving the following:
%
% min_x F(K*x) + G(x),
%
% where K is a known matrix, F(u) = a'*(log(u./a) + 1) and G(u) = sum(K*u).
%
%
% Required Parameters:
%     V:                non-negative given matrix (n x m)
%     W:                initial non-negative matrix factor (n x r)
%     H:                initial non-negative matrix factor (r x m)
%     N:                number of iterations (access to data)
%     D:                number of iterations for each ND problem
% 
% Output:
%     W:                optimal non-negative matrix factor
%     H:                optimal non-negative matrix factor
%     obj:              objective at each iteration (access to data)
%     time:             run time per iteration (access to data)
%
%
% Author: Felipe Yanez
% Copyright (c) 2014-2016

% Initialization
cputime0   = tic;
if ~isfield(options,'display')
        options.display = 0; 
end

timeerr=0; 
obj=[];
% Set parameters
chi   = -V./(W*H);
chi   = bsxfun(@times, chi, 1./max(bsxfun(@times, -W'*chi, 1./sum(W,1)')));
Wbar  = W;
Wold  = W;
Hbar  = H;
Hold  = H;
[n m] = size(V);
r     = size(H,1);
i=1;

time(i)=toc(cputime0);
time1=tic;
obj(i)  = KLobj(V,W,H); 
timeerr=toc(time1); % to remove the time of computing the objective 


while  i<= options.maxiter && time(i)<= options.timemax
   
    % Computation of H
    sigma = sqrt(n/r) * sum(W(:)) ./ sum(V,1)  / norm(W);
   % issparse(sigma)
   % pause
    tau   = sqrt(r/n) * sum(V,1)  ./ sum(W(:)) / norm(W);
    j=1; 
    while j <= D
        chi  = chi + bsxfun(@times, W*Hbar, sigma);
        chi  = (chi - sqrt(chi.^2 + bsxfun(@times, V, 4*sigma)))/2;
        H    = max(H - bsxfun(@times, W'*(chi + 1), tau), 0);
        Hbar = 2*H - Hold;
        Hold = H;
        j=j+1;
    end
    
    % Computation of W
    sigma = sqrt(m/r) * sum(H(:)) ./ sum(V,2)  / norm(H);
    tau   = sqrt(r/m) * sum(V,2)  ./ sum(H(:)) / norm(H);
    j=1; 
    while j <= D 
        chi  = chi + bsxfun(@times, Wbar*H, sigma);
        chi  = (chi - sqrt(chi.^2 + bsxfun(@times, V, 4*sigma)))/2;
        W    = max(W - bsxfun(@times, (chi + 1)*H', tau), 0);
        Wbar = 2*W - Wold;
        Wold = W;
        j=j+1;
    end
    
    % Objective and run time per iteration
    i=i+1;
     time1=tic;
     obj(i)  = KLobj(V,W,H); 
     timeerr=timeerr + toc(time1);
    time(i) = toc(cputime0) - timeerr;
    
    if  options.display ==1
     fprintf('PD: iteration %4d fitting error: %1.2e \n',i,obj(i));    
    end
    
        
end

end