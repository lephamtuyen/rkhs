function der = DLogPiDTheta(policy,s,u)

% derivative of w:
% compute mean
if (isfield(policy,'type'))
    phi = s;
else
    phi = getRBFFeatures(s);
end
mean= policy.w'*phi;

%dW
dW = (bsxfun(@times,(u-mean)*phi',1./(policy.sigma.^2)))';

dVar = -1./(policy.sigma) + ((u-mean).^2 ./ (policy.sigma.^3));

der = [dW(:);dVar];

end