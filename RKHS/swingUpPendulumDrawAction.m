function u = swingUpPendulumDrawAction(policy, s, n)
global A_max A_min

if (isfield(policy,'type') && strcmp(policy.type,'non-parametric') == 1)
    mean = getH(s, policy);
else
    phi = getRBFFeatures(s);
    mean= policy.w'*phi;
end
u = mvnrnd(mean,diag(policy.sigma.^2),n)';

% if (~isfield(policy,'type') || strcmp(policy.type,'non-parametric') == 0)
%     u(1,:) = min(u(1,:),A_max);
%     u(1,:) = max(u(1,:),A_min);
% end
end