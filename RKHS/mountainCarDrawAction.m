function u = mountainCarDrawAction(policy, s, n)
global A_max A_min
if (isfield(policy,'type') && strcmp(policy.type,'non-parametric') == 1)
    mean = getH(s, policy);
else
    phi = getRBFFeatures(s);
    mean= policy.w'*phi;
end
u = mvnrnd(mean,diag(policy.sigma.^2),n)';
end