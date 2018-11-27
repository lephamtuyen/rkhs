function value = kernelFunc(states1, states2)
global RBF

states_size = size(states1);
states1 = reshape(states1,[states_size(1), 1, states_size(2)]);
diffs = bsxfun(@minus, states1, states2);
value= exp(-0.5 * sum(bsxfun(@times,diffs.^2,1./(RBF.sigmas.^2)),1));

if (states_size(2)~=1)
    value = permute(value,[3,2,1]);
end

end