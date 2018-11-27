function phi = getRBFFeatures(x)
global RBF
diffs = bsxfun(@minus, RBF.centers, x);
phi = exp(-0.5 * sum(bsxfun(@times,diffs.^2,1./(RBF.sigmas.^2)),1));
phi = phi';
end