% this function finds KL divergence D(X,WH)
function dist = KLobj(X,W,H)


    if issparse(X)
        term1 = blockrecursivecompwiseprodsparselowrank(X,W,H,@xlogxdy); 
        dist = full(sum( term1(:) ) - sum(X(:)) + sum(sum( H.*( repmat(sum(W)',1,size(X,2)) ) ) )) ; 
    else
        Y = W*H; 
        Z = X.*log(X./(Y+eps) + eps) - X + Y; 
        dist = sum( Z(:) ); 
    end

    

end