function on_lr_rkhs
clear all; %#ok<CLFUN>
close all;
clc;

global lambda gamma domain A_dim numRuns MaxIter

domain = 'canon';
% domain = 'toy';
% domain = 'mountain_car';
% domain = '2_link_1_torque';
% domain = '2_link_2_torque';
% domain = 'drone_navigation';
% domain = 'ardupilot_navigation';
% domain = 'Swing_Pendulum';

initializeDomain;

% Discounted reward
gamma = 0.99;
rew = zeros(numRuns,MaxIter);
for i=1:numRuns
    fprintf('RUN: %d\n', i);
    
    % Initialize gaussian policy
    policy = initPolicy;
    
    % initial update counter
    e = 1;
    while (e <= MaxIter)
        % Take action, observe reward and next state
        data  = obtainData(policy);
        
        % Get Q approx
%         data = getQApprox(policy,data);
        
        % ///////////////////////////// RKHS //////////////////////////////////
        policy_new = RKHS(policy, data);
        
        % ///////////////////////////// Line Search ///////////////////////////
        lambda = findTheBestLearningRateUsingLineSearch(policy_new, policy);
%         lambda = 0.1;
        % ///////////////////////////// Update policy /////////////////////
        policy_new.params(1:A_dim,size(policy.params,2)+1:end) = lambda*policy_new.params(1:A_dim,size(policy.params,2)+1:end);
%         policy_new.params(1:A_dim,:) = lambda*policy_new.params(1:A_dim,:);
        policy = policy_new;
        % ///////////////////////////// Update policy /////////////////////
        
        % remove first row
        if(e==1)
            policy.params = policy.params(:,2:end);
        end
        
        % Sparsification
        policy = Sparsification(policy);
        rew(i,e) = data.averageReward;
        
        printLog(e, policy, data);
        
        % Increase
        e = e + 1;
    end
end
filename = strcat('data/',domain,'_on_lr_rkhs');
save(filename,'rew');
end

function new_policy = Sparsification(policy)
global A_dim

Kernel = kernelFunc(policy.params(A_dim+1:end,:),policy.params(A_dim+1:end,:));
Residue = getH(policy.params(A_dim+1:end,:), policy);

new_policy = KernelMatchingPursuit(policy,Residue,Kernel,0.001);
end

function Approx = KMP(Residue,Kernel,tol)

kernel_size = size(Kernel,2);
ADDED = zeros(kernel_size,1);
PHI = zeros(kernel_size,kernel_size);
weights =zeros(kernel_size,1);
OUTPUT = Residue;
for i=1:kernel_size
    remain_idxs = find(ADDED==0);
    remain_phi = Kernel(:,remain_idxs');
    normm1 = Residue*remain_phi;
    normm2 = sum(remain_phi.*remain_phi,1);
    
%     nominator = (nthroot(sum(normm1.^(redidue_dim),1),redidue_dim)).^2;
%     denominator = normm2;
    
    nominator = sqrt(sum(normm1.^2,1));
    denominator = sqrt(normm2);
    
    [~,best_index] = max(nominator./denominator);
    weight = normm1(:,best_index)/normm2(best_index);
    weights(i) = weight;
    
    % Back-fitting
    ADDED(remain_idxs(best_index)) = 1; % Added
%     Residue = Residue - weight*(remain_phi(:,best_index))';
    PHI(:,i) = remain_phi(:,best_index);
    
%     num_phi = numel(find(ADDED==1));
%     NEW_W = pinv(PHI(:,1:num_phi)'*PHI(:,1:num_phi)+eye(num_phi)*1.e-10)*PHI(:,1:num_phi)'*OUTPUT';
%     Approx = (PHI(:,1:num_phi)*NEW_W)';
%     Residue = OUTPUT - Approx;
    
%     norm_residue = norm(Residue);
%     if (norm_residue < tol)
%         break;
%     end
end

num_phi = numel(find(ADDED==1));
NEW_W = pinv(PHI(:,1:num_phi)'*PHI(:,1:num_phi)+eye(num_phi)*1.e-10)*PHI(:,1:num_phi)'*OUTPUT';
Approx = (PHI(:,1:num_phi)*NEW_W)';

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
global lambda_range A_dim evel_episodes

best_alpha = 0.0;
best_value = -1000000000;
numEval = 40;
% for grid = lambda_range(1):(diff(lambda_range)/numEval):lambda_range(2)
for grid = 1:numEval
    alpha_new = 0.01 * (((1.1)^(grid-1))-1) + 0.0000001;
    %     alpha_new = grid;
    temp_policy = policy_new;
    temp_policy.params(1:A_dim,size(policy.params,2)+1:end) = alpha_new*temp_policy.params(1:A_dim,size(policy.params,2)+1:end);
%     temp_policy.params(1:A_dim,:) = alpha_new*temp_policy.params(1:A_dim,:);
    
    % Data with new policy
    data_new  = obtainData(temp_policy,evel_episodes);
    if (data_new.averageReward > best_value)
        best_value = data_new.averageReward;
        best_alpha = alpha_new;
    end
end
end

function new_policy = RKHS(policy, data)
global A_dim S_dim gamma

new_policy = policy;
policy_size = size(policy.params,2);
new_policy_size = policy_size+sum(data.length);
new_policy.params = zeros(A_dim+S_dim,new_policy_size);
new_policy.params(:,1:policy_size) = policy.params;
i=policy_size+1;
idx = 1;

for epi = 1 : data.numOfEpisodes
    Steps = data.length(epi);
    s = data.episodes(epi).x;
    a = data.episodes(epi).u;
    hs = getH(data.episodes(epi).x(:,1:data.length(epi)),policy);
    gammas = gamma.^(0:1:Steps-1);
    R = sum(bsxfun(@times,data.episodes(epi).r,gammas));
    dLogPi = pinv(diag(policy.sigma.^2))*(a - hs);

    grad = bsxfun(@times, R/data.numOfEpisodes,dLogPi);
%     grad = bsxfun(@times, data.Q_approx(idx:idx+Steps-1)/data.numOfEpisodes,dLogPi);
    
    new_policy.params(:,i:i+Steps-1)=[grad;s];
    i = i + Steps;idx=idx+Steps;
end
end

function data = obtainData(policy, episodes, steps)
global Steps Episodes gamma

if (nargin == 3)
    n_episodes = episodes;
    n_steps = steps;
elseif (nargin == 2)
    n_episodes = episodes;
    n_steps = Steps;
else
    n_episodes = Episodes;
    n_steps = Steps;
end

data.averageReward = 0;
data.numOfEpisodes = n_episodes;
data.length = zeros(n_episodes,1);

% Perform episodes
for epi=1:n_episodes
    % Draw the first state
    data.episodes(epi).x(:,1) = drawStartState;
    
    % Perform an episode of length Steps
    for step=1:n_steps
        % Draw an action from the policy
        data.episodes(epi).u(:,step) = drawAction(policy, data.episodes(epi).x(:,step),1);
        
        % Obtain the reward from the
        data.episodes(epi).r(step) = rewardFnc(data.episodes(epi).x(:,step), data.episodes(epi).u(:,step));
        
        % Draw next state from environment
        [x_next, isValidNextStep] = drawNextState(data.episodes(epi).x(:,step), data.episodes(epi).u(:,step));
        
        if (isValidNextStep == 0 || step == n_steps)
            break;
        end
        
        data.episodes(epi).x(:,step+1) = x_next;
    end
    
    data.length(epi) = step;
    gammas = gamma.^(0:1:step-1);
    data.averageReward = data.averageReward + sum(bsxfun(@times,data.episodes(epi).r,gammas),2);
end

data.averageReward = data.averageReward/n_episodes;
end

function Q = getQ(policy,s0,a0,steps)
global gamma

% Draw the first state
s = s0;
a=a0;
r = rewardFnc(s,a);
Q = r;
% Perform an episode of length Steps
for step=2:steps
    % Draw next state from environment
    [s, isValidNextStep] = drawNextState(s, a);
    
    if (isValidNextStep == 0)
        break;
    end
    
    % Draw an action from the policy
    a = drawAction(policy,s,1);
    
    % Obtain the reward from the
    r = rewardFnc(s,a);
    Q = Q + gamma^(step-1)*r;
end
end

% Using kernel matching pursuit to find weight factor
function data = getQApprox(policy, data)
global Steps

sample_size = sum(data.length);
Q = zeros(1,sample_size);
Kernel = zeros(sample_size,sample_size);

s = horzcat(data.episodes.x);
a = horzcat(data.episodes.u);

for i = 1 : sample_size
    si = s(:,i);
    ai = a(:,i);
    hsi = getH(s(:,i),policy);
    aj = a;
    hsj = getH(s, policy);
    
    ker = kernelFunc(si,s);
    inverse_var = pinv(diag(policy.sigma.^2));
    Kernel(i,:) = sum((inverse_var*(ai-hsi)*ker).*(inverse_var*(aj-hsj)),1);
    Q(i) = getQ(policy,si,ai,Steps);
end

data.Q_approx = KMP(Q,Kernel,0.001);

end

