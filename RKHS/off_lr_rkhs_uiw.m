function off_lr_rkhs_uiw
clear all; %#ok<CLFUN>
close all;
clc;

global lambda gamma domain A_dim numRuns MaxIter Steps Episodes

% domain = 'canon';
% domain = 'toy';
% domain = 'Swing_Pendulum';
% domain = 'mountain_car';
% domain = '2_link_1_torque';
% domain = '2_link_2_torque';
domain = 'drone_navigation';

initializeDomain;

% Discounted reward
gamma = 0.99;
rew = zeros(numRuns,MaxIter);
for i=1:numRuns
    fprintf('RUN: %d\n', i);
    
    % Initialize gaussian policy
    policy = initPolicy;
    % initial update counter
    e = 1;data=[];
    while (e <= MaxIter)
        % Take action, observe reward and next state
        data = obtainData(policy, data);

        % ///////////////////////////// RKHS //////////////////////////////////
        policy_new = RKHS(policy, data);

        % ///////////////////////////// Line Search ///////////////////////////
        lambda = findTheBestLearningRateUsingLineSearch(policy_new, policy);

        % ///////////////////////////// Update policy /////////////////////
        policy_new.params(1:A_dim,size(policy.params,2)+1:end) = lambda*policy_new.params(1:A_dim,size(policy.params,2)+1:end);
        policy = policy_new;
        % ///////////////////////////// Update policy /////////////////////
        
        % remove first row
        if(e==1)
            policy.params = policy.params(:,2:end);
        end
        
        % Sparsification
        policy = Sparsification(policy);
        
        evaluate = getTrajectory(policy, Episodes, Steps);
        rew(i,e) = evaluate.averageReward;
        
        printLog(e, policy, evaluate);
        
        % Increase
        e = e + 1;
    end
end

filename = strcat('data/',domain,'_off_lr_rkhs_uiw');
save(filename,'rew');
end

function new_policy = Sparsification(policy)
global A_dim

Kernel = kernelFunc(policy.params(A_dim+1:end,:),policy.params(A_dim+1:end,:));
Residue = getH(policy.params(A_dim+1:end,:), policy);

new_policy = KernelMatchingPursuit(policy,Residue,Kernel,0.001);
end

function new_policy = KernelMatchingPursuit(policy,Residue,Kernel,tol)
global A_dim RBF tol_dist

new_policy = policy;new_policy.params=[];
policy_size = size(policy.params,2);
ADDED = zeros(policy_size,1);
PHI = zeros(policy_size,RBF.numOfCenters);

OUTPUT = Residue;
for i=1:policy_size
    remain_idxs = find(ADDED==0);
    remain_phi = Kernel(:,remain_idxs');
    normm1 = Residue*remain_phi;
    normm2 = sum(remain_phi.*remain_phi,1);
    
    nominator = sqrt(sum(normm1.^2,1));
    denominator = sqrt(normm2);
    
    [~,best_index] = max(nominator./denominator);
    weight = normm1(:,best_index)/normm2(best_index);
    
    % Back-fitting
    new_center = policy.params(:,remain_idxs(best_index));

    ADDED(remain_idxs(best_index)) = 1; % Added
    Residue = Residue - weight*(remain_phi(:,best_index))';
    PHI(:,i) = remain_phi(:,best_index);
    new_policy.params=horzcat(new_policy.params,new_center);

    if (size(new_policy.params,2) == RBF.numOfCenters)
        break;
    end
end

num_phi = numel(find(ADDED==1));
NEW_W = pinv(PHI(:,1:num_phi)'*PHI(:,1:num_phi)+eye(num_phi)*1.e-10)*PHI(:,1:num_phi)'*OUTPUT';
new_policy.params(1:A_dim,:) = NEW_W';

end


function best_alpha = findTheBestLearningRateUsingLineSearch(policy_new, policy)
global lambda_range A_dim evel_episodes Episodes Steps

best_alpha = 0.0;
best_value = -1000000000;
numEval = 40;
% for grid = lambda_range(1):(diff(lambda_range)/numEval):lambda_range(2)
for grid = 1:numEval
    alpha_new = 0.01 * (((1.1)^(grid-1))-1) + 0.0000001;
%     alpha_new = grid;

    temp_policy = policy_new;
    temp_policy.params(1:A_dim,size(policy.params,2)+1:end) = alpha_new*temp_policy.params(1:A_dim,size(policy.params,2)+1:end);
    
    % Data with new policy
%     data_new  = obtainData(temp_policy,[],evel_episodes);
    data_new = getTrajectory(temp_policy, Episodes, Steps);
    if (data_new.averageReward > best_value)
        best_value = data_new.averageReward;
        best_alpha = alpha_new;
    end
end
end

function data = find_optimal_sampling_policy(data)

if (size(data,1) <= 2)
    return;
end

% Q = zeros(size(data,1)-1);
% l = zeros(size(data,1)-1,1);
% for i = 1:size(data,1)-1
%     for j = i:size(data,1)-1
%         Q(i,j) = expectation_between_2_policy(data(i).data,data(j).data);
%         Q(j,i) = Q(i,j);
%     end
%     l(i) = expectation_between_2_policy(data(end).data,data(i).data);
% end
% 
% H = (Q + 1e-10*eye(size(data,1)-1));
% Aeq = ones(1,size(data,1)-1);
% beq = 1;
% lb = zeros(size(data,1)-1,1);
% options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
% optimal_policy_coefficients = quadprog(H,l,[],[],Aeq,beq,lb,[],[],options)
% for i = 1:size(data,1)-1
%     data(i).coef = optimal_policy_coefficients(i);
% end

% returns = horzcat(data.averageReward);
% for i = 1:size(data,1)-1
%     data(i).coef = data(i).averageReward/sum(returns(1:size(data,1)-1));
% end

% coef = 1/(size(data,1)-1);
% for i = 1:size(data,1)-1
%     data(i).coef = coef;
% end

end

function Q = expectation_between_2_policy(x1,x2)
global A_dim

x1_x = x1(A_dim+1:end,:);
x2_x = x2(A_dim+1:end,:);
x1_y = reshape(x1(1:A_dim,:),A_dim,1,size(x1,2));
x2_y = x2(1:A_dim,:);

ker = kernelFunc(x1_x,x2_x);
ker_size = size(ker);
reshape_ker = reshape(ker',[1,ker_size(2),ker_size(1)]);
Q = sum(bsxfun(@times,bsxfun(@times,reshape_ker,x2_y),x1_y),1);
Q = sum(sum(sum(Q)))/(size(x1,2)*size(x2,2));
end

function [new_policy] = RKHS(policy, data)
global A_dim S_dim gamma

new_policy = policy;
new_policy.id = getPolicyID;
policy_size = size(policy.params,2);
new_policy_size = policy_size+sum(vertcat(data.length));
new_policy.params = zeros(A_dim+S_dim,new_policy_size);
new_policy.params(:,1:policy_size) = policy.params;
i=policy_size+1;
idx = 1;
numOfEpisodes = sum(vertcat(data.numOfEpisodes));

for poli=1:size(data,1)
    for epi = 1 : data(poli).numOfEpisodes
        Steps = data(poli).length(epi);
        gammas = gamma.^(0:1:Steps-1);
        s = data(poli).episodes(epi).x;
        a = data(poli).episodes(epi).u;
        R = sum(bsxfun(@times,data(poli).episodes(epi).r,gammas));
        hs = getH(data(poli).episodes(epi).x(:,1:data(poli).length(epi)),policy);
        dLogPi = pinv(diag(policy.sigma.^2))*(a - hs);
        
        epi_policy = data(poli).policy;
        iw = 1;
        % Update importance weight
        if (policy.id ~= epi_policy.id)
            for step = 1:Steps
                si = data(poli).episodes(epi).x(:,step);
                ai = data(poli).episodes(epi).u(:,step);
                
                nom = mvnpdf(ai',(getH(si,policy))',diag(policy.sigma.^2));
                den = mvnpdf(ai',(getH(si,epi_policy))',diag(epi_policy.sigma.^2));

                currentIW = nom/den;
                
                
                iw = iw * currentIW;
                if(isnan(iw))
                    iw=0;
                end
                if (isinf(iw))
                    iw=1;
                end
            end
        end
        
        grad = bsxfun(@times,iw*R/numOfEpisodes,dLogPi);
        
        new_policy.params(:,i:i+Steps-1)=[grad;s];
        i = i + Steps;idx=idx+Steps;
    end
end
end

function new_data = getTrajectory(policy,n_episodes,n_steps)
global gamma
new_data.coef = 1;
new_data.numOfEpisodes = n_episodes;
new_data.averageReward = 0;
new_data.policy = policy;
new_data.length = zeros(n_episodes,1);
% Perform episodes
for epi=1:n_episodes
    
    % Draw the first state
    new_data.episodes(epi).x(:,1) = drawStartState;
    
    % Perform an episode of length Steps
    for step=1:n_steps
        % Draw an action from the policy
        new_data.episodes(epi).u(:,step) = drawAction(policy, new_data.episodes(epi).x(:,step),1);
        
        % Obtain the reward from the
        new_data.episodes(epi).r(step) = rewardFnc(new_data.episodes(epi).x(:,step), new_data.episodes(epi).u(:,step));
        
        % Draw next state from environment
        [x_next, isValidNextStep] = drawNextState(new_data.episodes(epi).x(:,step), new_data.episodes(epi).u(:,step));
        
        if (isValidNextStep == 0 || step == n_steps)
            break;
        end
        
        new_data.episodes(epi).x(:,step+1) = x_next;
    end
    
    new_data.length(epi) = step;
    gammas = gamma.^(0:1:step-1);
    new_data.averageReward = new_data.averageReward + sum(bsxfun(@times,new_data.episodes(epi).r,gammas),2);
end
new_data.averageReward = new_data.averageReward/n_episodes;
a = horzcat(new_data.episodes.u);
s = horzcat(new_data.episodes.x);
new_data.data = [a;s];
end

function new_data = obtainData(policy, old_data, episodes, steps)
global Steps Episodes

if (nargin == 4)
    n_episodes = episodes;
    n_steps = steps;
elseif (nargin == 3)
    n_episodes = episodes;
    n_steps = Steps;
else
    n_episodes = Episodes;
    n_steps = Steps;
end

new_data = getTrajectory(policy, n_episodes, n_steps);
new_data = [old_data;new_data];

% if (size(new_data,1) > 5)
%     coefs = horzcat(new_data.coef);
%     [~,idx] = min(coefs(1:size(new_data,1)-1));
%     idx
%     new_data(idx) = [];
% end

% if (size(new_data,1) > 5)
%     returns = horzcat(new_data.averageReward);
%     [~,idx] = min(returns(1:size(new_data,1)-1));
%     idx
%     new_data(idx) = [];
% end

if (size(new_data,1) > 5)
    new_data = new_data(end-4:end);
end

end

function k = getCompatibleKernelMatrix(s1, s2, policy)
global A_dim

k = zeros(size(s1,2),size(s2,2));
inverse_var = pinv(diag(policy.sigma.^2));
sj = s2(A_dim+1:end,:);
aj = s2(1:A_dim,:);
hsj = getH(sj,policy);
exp2 = inverse_var*(aj-hsj);
for i=1:size(s1,2)
    si = s1(A_dim+1:end,i);
    ai = s1(1:A_dim,i);
    hsi = getH(si,policy);
    ker = kernelFunc(si,sj);
    exp1 = bsxfun(@times,ker,inverse_var*(ai-hsi));
    k(i,:) = sum(bsxfun(@times,exp1,exp2),1);
end
end

function ErrorEstimation = ApproximationErrorEstimation(policy,data,v)
fold = 5;

G = zeros(fold,1);
numOfEpisodes_fold = data.numOfEpisodes/fold;
for i = 1:fold %fold
    data_fold = data;
    index = true(1, size(data_fold.episodes, 2));
    index((i-1)*numOfEpisodes_fold+1:i*numOfEpisodes_fold) = false;
    data_fold.episodes = data_fold.episodes(index);
    data_fold.length = data_fold.length(index);
    data_fold.numOfEpisodes = data_fold.numOfEpisodes-numOfEpisodes_fold;
  
    % Perform sparsification
    D = ALDSparsification(policy, data_fold);
    
    s = horzcat(data_fold.episodes.x);
    a = horzcat(data_fold.episodes.u);
    Kernel = getCompatibleKernelMatrix([a;s], D, policy);
    w = getParameterKLSTDQ(policy,data_fold,Kernel,v);
    Q_approx = Kernel*w;
    G(i) = getSquaresErrorApprox(policy, data_fold, Q_approx, v);
end
ErrorEstimation = mean(G);
end

function ErrorEstimation = ApproximationErrorEstimation_simple(policy,data,flat)
s = horzcat(data.episodes.x);
a = horzcat(data.episodes.u);

% Perform sparsification
D = ALDSparsification(policy, data);
    
Kernel = getCompatibleKernelMatrix([a;s], D, policy);
w = getParameterKLSTDQ(policy,data,Kernel,flat);
Q_approx = Kernel*w;
ErrorEstimation = getSquaresErrorApprox(policy, data, Q_approx, flat);

end

function G = getSquaresErrorApprox(policy, data, Q_approx, v)
global gamma

i = 1;G = 0;numofpolicy = 1;
for epi = 1 : data.numOfEpisodes
    iw=1;g = 0;
    for step=1:data.length(epi)
        r = data.episodes(epi).r(step);
        si = data.episodes(epi).x(:,step);
        ai = data.episodes(epi).u(:,step);
        epi_policy = data.episodes(epi).policy;
        if (~isequal(policy.params,epi_policy.params))
            numofpolicy = numofpolicy + 1;
        end
        
        % Update importance weight
        nom = mvnpdf(ai',(getH(si,policy))',diag(policy.sigma.^2));
        den = mvnpdf(ai',(getH(si,epi_policy))',diag(epi_policy.sigma.^2));
        currentIW = nom/den;
        iw = iw * currentIW;
        if(isnan(iw))
            iw=0;
        end
        if (isinf(iw))
            iw=1;
        end
        if (step == data.length(epi))
            g = g + ((iw^v)*(Q_approx(i) - r))^2;
        else
            g = g + ((iw^v)*(Q_approx(i) - r - gamma*Q_approx(i+1)))^2;
        end
    end
    
    G = G + g / data.length(epi);
end

G = G / (data.numOfEpisodes*numofpolicy);

end

function D = ALDSparsification(policy, data)
% global A_dim S_dim ALD

s = horzcat(data.episodes.x);
a = horzcat(data.episodes.u);
D = [a;s];

% sample_size = sum(data.length);
% D=zeros(A_dim+S_dim,sample_size);idx=0;
% for epi = 1 : data.numOfEpisodes
%     for step=1:data.length(epi)
%         si = data.episodes(epi).x(:,step);
%         ai = data.episodes(epi).u(:,step);
% 
%         if (idx == 0)
%             idx = 1;
%             D(:,idx)=[ai;si];
%         else
%             K = getCompatibleKernelMatrix(D(:,1:idx), D(:,1:idx), policy);
%             k = getCompatibleKernelMatrix([ai;si], D(:,1:idx), policy);
%             ktt = getCompatibleKernelMatrix([ai;si], [ai;si], policy);
%             
%             ct = pinv(K)*k';
%             sigt = ktt-k*ct;
%             if(sigt > ALD)
%                 idx = idx+1;
%                 D(:,idx)=[ai;si];
%             end
%         end
%     end
% end
% D = D(:,1:idx);
end

function [w, iw] = getParameterKLSTDQ(policy,data,Kernel,flat)
global gamma

% klstd-q
A=0;b=0;
i=1;
iw = zeros(1,sum(data.length));
for epi = 1 : data.numOfEpisodes
    % first action state
    r = data.episodes(epi).r(1);
    
    k = Kernel(i,:);
    i = i + 1;
    iw_i = 1.0;
    for step=1:data.length(epi)
        si = data.episodes(epi).x(:,step);
        ai = data.episodes(epi).u(:,step);
        epi_policy = data.episodes(epi).policy;
        
        if (flat ~= 0)
            % Update importance weight
            nom = mvnpdf(ai',(getH(si,policy))',diag(policy.sigma.^2));
            den = mvnpdf(ai',(getH(si,epi_policy))',diag(epi_policy.sigma.^2));
            currentIW = nom/den;
            iw_i = iw_i * currentIW;
            if(isnan(iw_i))
                iw_i=0;
            end
            if (isinf(iw_i))
                iw_i=1;
            end
        end
        
        iw(i-1) = iw_i;
        
        if (step == data.length(epi)) % last step
            A = A + (iw_i^flat)*(k'*k)  + 0.000000000001;
            b = b + (iw_i^flat)*(k'*r);
        else
            rn = data.episodes(epi).r(step+1);
            kn = Kernel(i,:);
            i = i + 1;

            A = A + (iw_i^flat)*(k'*(k-gamma*kn)) + 0.000000000001;
            b = b + (iw_i^flat)*(k'*r);

            r = rn;
            k = kn;
        end

%         if (any(isnan(A)) | any(isinf(A)) | any(isnan(b)) | any(isinf(b)))
%             dsdhsd=0;
%         end
    end
end
w = pinv(A)*b;
end

% Using kernel matching pursuit to find weight factor
function data = getLSTDQApprox(policy,data,v)

% Perform sparsification
D = ALDSparsification(policy, data);

s = horzcat(data.episodes.x);
a = horzcat(data.episodes.u);
Kernel = getCompatibleKernelMatrix([a;s], D, policy);
[w,iw] = getParameterKLSTDQ(policy,data,Kernel,v);
data.Q_approx = (Kernel*w)';
data.iw = iw;
end