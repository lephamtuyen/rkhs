function hs = getH(states, policy)
% eval: h(s) = sum_i \alpha_i K(s_i,s)
global A_dim A_max A_min

yi = policy.params(1:A_dim,:);
ker_matrix = kernelFunc(states,policy.params(A_dim+1:end,:));
ker_size = size(ker_matrix);
reshape_ker = reshape(ker_matrix',[1,ker_size(2),ker_size(1)]);
hs = sum(bsxfun(@times,reshape_ker,yi),2);

if (ker_size(1)~=1)
    hs = permute(hs,[1,3,2]);
end

for i=1:size(hs,1)
    hs(i,:) = min(hs(i,:),A_max(i));
    hs(i,:) = max(hs(i,:),A_min(i));
end


end