% Given the sparse matrix X, and the low-rank factors (W,H), 
% Compute  Y = X .* (W*H).^beta
% 
% This is the component-wise product between the sparse matrix X and the
% low-rank matrix WH component-wise exponentiated by beta.
% Since W*H can be dense, one should not compute W*H explicitely. 

function Y = compwiseprodsparselowrank(X,W,H,fun) 

[i,j,s] = find(X);
Y = sparse(size(X,1), size(X,2)); 
N = length(s); 
%fprintf('Number of non-zero entries: %2.0d. \n', N); 
for t = 1 : N
    Y(i(t),j(t)) = fun( X(i(t),j(t)) , (W(i(t),:)*H(:,j(t))) ); 
%     if mod(t,1e4) == 0
%         fprintf('%2.0d...', t);
%     end
%     if mod(t,1e5) == 0
%         fprintf('\n');
%     end
end