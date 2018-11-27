function pg_gpomdb
clear all; %#ok<CLFUN>
close all;
clc;

global lambda gamma domain A_dim RBF lb ub numParams

domain = 'canon';
% domain = '2_link_1_torque';
% domain = '2_link_2_torque';

initializeDomain;

% Discounted reward
gamma = 1;

% Initialize gaussian policy
policy = initGaussianPolicy;

% initial update counter
e = 1; MaxIter = 1000000;maxvalue = -1000000000;
y = zeros(MaxIter,1);

numParams = A_dim*RBF.numFeatures+A_dim;
params = zeros(numParams,MaxIter);

while (e < MaxIter)
    % Parameters
    theta = [policy.w(:);policy.sigma];
    params(:,e)=theta;
    
    % Take action, observe reward and next state
    data  = obtainData(policy);
    
    % draw
%     draw = 1;
%     if (draw==1)
%         drawDoubleLink(data.episodes(1),e);
%     end
    
    % ///////////////////////////// NaturalActorCritic /////////////////////
%     dTheta = episodicGPOMDPWithBaseline(policy, data);
    dTheta = GPOMDP(policy, data);
    % ///////////////////////////// Line Search ///////////////////////////
%     lambda = findTheBestLearningRateUsingLineSearch(dTheta, theta, policy, data);
    
    % ///////////////////////////// Update policy /////////////////////
    theta_new=kk_proj(theta + lambda * dTheta,ub,lb);
    policy.w = reshape(theta_new(1:A_dim*RBF.numFeatures),[RBF.numFeatures, A_dim]);
    policy.sigma = reshape(theta_new(A_dim*RBF.numFeatures+1:A_dim*RBF.numFeatures+A_dim),[A_dim, 1]);
    % ///////////////////////////// Update policy /////////////////////
    
    y(e) = data.averageReward;
    
%     if (maxvalue < data.averageReward)
%         maxvalue = data.averageReward;
%         filename = strcat('data/',int2str(e));
%         save(filename,'data');
%     end
    
    if (mod(e,100)==0)
        save('data/params','params');
        save('data/y','y');
    end
    
    printLog(e, policy, data);
    
    % Increase
    e = e + 1;
end

end

function printLog(e, policy, data)
global domain
switch domain
    case 'canon'
        fprintf('step: %d, reward: %f\n', e, data.averageReward);
        fprintf('sigma: %f\n', policy.sigma);
    case '2_link_1_torque'
        fprintf('step: %d, reward: %f, steps: %f\n', e, data.averageReward, mean(data.length));
        fprintf('velocity max 1: %f, velocity max 2: %f\n', max(data.episodes(1).x(2,:)), max(data.episodes(1).x(4,:)));
        fprintf('velocity min 1: %f, velocity min 2: %f\n', min(data.episodes(1).x(2,:)), min(data.episodes(1).x(4,:)));
        fprintf('actionmax1: %f, actionmin1: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('actionmax2: %f, actionmin2: %f\n', max(data.episodes(1).u(2,:)),min(data.episodes(1).u(2,:)));
        fprintf('sigma: %f\n', policy.sigma);
    case '2_link_2_torque'
        fprintf('step: %d, reward: %f, steps: %f\n', e, data.averageReward, mean(data.length));
        fprintf('velocity max 1: %f, velocity max 2: %f\n', max(data.episodes(1).x(2,:)), max(data.episodes(1).x(4,:)));
        fprintf('velocity min 1: %f, velocity min 2: %f\n', min(data.episodes(1).x(2,:)), min(data.episodes(1).x(4,:)));
        fprintf('actionmax1: %f, actionmin1: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('actionmax2: %f, actionmin2: %f\n', max(data.episodes(1).u(2,:)),min(data.episodes(1).u(2,:)));
        fprintf('sigma: %f\n', policy.sigma);
end

end

function lambda = findTheBestLearningRateUsingLineSearch(dTheta, theta, policy, data)
global ub lb RBF A_dim lambda_min
MAXXX = data.averageReward;
step_size_UB = 1000.0;
idxs = ((dTheta > 0) & (theta > 0));
temp = theta(idxs)./dTheta(idxs);
step_size_UB = min([temp; step_size_UB]);
lambda = lambda_min;
for step = 0:3:150
    step_size = 0.001 * (1.1)^step;
    if(step_size > step_size_UB)
        break;
    end
    
    theta_new=kk_proj(theta + step_size * dTheta,ub,lb);
    policy_new = policy;
    policy_new.w = reshape(theta_new(1:A_dim*RBF.numFeatures),[RBF.numFeatures, A_dim]);
    policy_new.sigma = reshape(theta_new(A_dim*RBF.numFeatures+1:A_dim*RBF.numFeatures+A_dim),[A_dim, 1]);

    % Data with new policy
    data_new  = obtainData(policy_new);
    if (data_new.averageReward > MAXXX)
        MAXXX = data_new.averageReward;
        lambda = step_size;
    end
end
end

function px = kk_proj(x,ub,lb)
px=min(ub,x);
px=max(lb,px);
end

function dJdtheta = GPOMDP(policy, data)
global gamma numParams

i = 0; j = 0;
dJdtheta = zeros(numParams,1);
for epi = 1 : data.numOfEpisodes
    dSumPi = zeros(numParams,1);
    for step = 1 : data.length(epi)
        dSumPi = dSumPi + DLogPiDTheta(policy, data.episodes(epi).x(:,step), data.episodes(epi).u(step));
        rew    = gamma^(step-1)*data.episodes(epi).r(step);
        dJdtheta = dJdtheta + dSumPi * rew;
        j = j + 1;
    end;
    i = i + 1;
end;

if(gamma==1)
    dJdtheta = dJdtheta / j;
else
    dJdtheta = (1-gamma)*dJdtheta / data.numOfEpisodes;
end
end

function grad = episodicGPOMDPWithBaseline(policy, data)
global gamma numParams

sumDlog  = zeros(numParams,max(data.length),data.numOfEpisodes);
rewgamma = zeros(data.numOfEpisodes,max(data.length));

bnum = zeros(numParams, max(data.length));
bden = zeros(numParams, max(data.length));
for epi = 1 : data.numOfEpisodes
    der = zeros(numParams, data.length(epi));
    for step = 1 : data.length(epi)
        der(:,step) = DLogPiDTheta(policy, data.episodes(epi).x(:,step), data.episodes(epi).u(:,step));
    end
    
    gammas = gamma.^(0:1:step-1);
    sumDlog(:,1:step,epi) = cumsum(der,2);
    sumDlog2 = cumsum(der,2).^2;
    rewgamma(epi,1:step) = bsxfun(@times,data.episodes(epi).r,gammas);
    bden(:,1:step) = bden(:,1:step) + sumDlog2;
    bnum(:,1:step) = bnum(:,1:step) + bsxfun(@times,sumDlog2,rewgamma(epi,1:step));
end

% baseline
b = bnum ./ bden;
b(isnan(b)) = 0;

% gradient
grad = zeros(numParams,1);
for epi = 1 : data.numOfEpisodes
    steps = data.length(epi);
    grad = grad + sum(bsxfun(@times, sumDlog(:,1:steps,epi), bsxfun(@plus,rewgamma(epi,1:steps),-b(:,1:steps))),2);
end

if(gamma==1)
    grad = grad / sum(data.length);
else
    grad = (1-gamma)*grad / data.numOfEpisodes;
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
    temp = 0;
    for step=1:n_steps
        % Draw an action from the policy
        data.episodes(epi).u(:,step) = drawAction(policy, data.episodes(epi).x(:,step),1);
        
        % Draw next state from environment
        [data.episodes(epi).x(:,step+1), isValidNextStep] = drawNextState(data.episodes(epi).x(:,step), data.episodes(epi).u(:,step));
        
        % Obtain the reward from the
        data.episodes(epi).r(step) = rewardFnc(isValidNextStep, step, data.episodes(epi).x(:,step), data.episodes(epi).u);
        
        temp = temp + gamma^(step-1)*data.episodes(epi).r(step);
        
        if (isValidNextStep == 0)
            break;
        end
    end
    
    data.length(epi) = step;
    
    if(gamma==1)
        temp = temp/step;
    else
        temp = (1-gamma)*temp;
    end
    
    data.averageReward = data.averageReward + temp;
end

data.averageReward = data.averageReward/n_episodes;
end

function x0 = drawStartState
global domain
switch domain
    case 'canon'
        x0 = canonDrawState;
    case '2_link_1_torque'
        x0 = twoLinkOneTorqueDrawStartState;
    case '2_link_2_torque'
        x0 = twoLinkTwoTorqueDrawStartState;
end
end

function [xn, valid] = drawNextState(x,u)
global domain
switch domain
    case 'canon'
        xn = canonDrawState;
        valid = 1;
    case '2_link_1_torque'
        [xn, valid] = twoLinkOneTorqueDrawNextState(x,u);
    case '2_link_2_torque'
        [xn, valid] = twoLinkTwoTorqueDrawNextState(x,u);
end
end

function u = drawAction(policy, x, n)
global domain
switch domain
    case 'canon'
        u = canonDrawAction(policy, x, n);
    case '2_link_1_torque'
        u = twoLinkOneTorqueDrawAction(policy, x, n);
    case '2_link_2_torque'
        u = twoLinkTwoTorqueDrawAction(policy, x, n);
end
end

function rew = rewardFnc(valid, step, x, u)
global domain
switch domain
    case 'canon'
        rew = canonRewardFnc(x,u(:,step));
    case '2_link_1_torque'
        rew = twoLinkOneTorqueRewardFnc(valid, step, x, u);
    case '2_link_2_torque'
        rew = twoLinkTwoTorqueRewardFnc(valid, step, x, u);
end
end

function reward = averageReward(data)
global gamma
reward = 0;
for trial = 1 : size(data, 2)
    temp = 0;
    for step = 1 : size(data(trial).r, 2)
        temp = temp + gamma^(step-1)*data(trial).r(step);
    end
    
    if(gamma==1)
        temp = temp/size(data(trial).r, 2);
    else
        temp = (1-gamma)*temp;
    end
    
    reward = reward + temp;
end

reward = reward/size(data,2);

end

function initializeRBF(dimens, dimensions, partitions)
global RBF S_max S_min

RBF.dimens = dimens;
RBF.dimensions = dimensions;
RBF.partitions = partitions;
RBF.numFeatures = prod(RBF.partitions);
RBF.sigmas = (S_max-S_min)./(2.0*RBF.partitions);
RBF.centers = linspace(S_min(1)+RBF.sigmas(1),S_max(1)-RBF.sigmas(1),RBF.partitions(1));
for i=2:RBF.dimens
    temp = linspace(S_min(i)+RBF.sigmas(i),S_max(i)-RBF.sigmas(i),RBF.partitions(i));
    RBF.centers = combvec(RBF.centers,temp);
end
end

function initializeDomain
global lambda_min A_max A_min evaluateEpisodes S_max S_min A_dim S_dim domain Steps Episodes RBF lb ub basis_dim lambda rbf_centers rbf_sigma

switch domain
    case 'canon'
        % Cannon Toy task
        % State (d, w):  (distance, wind)
        % Action (alpha, v): (alpha, velocity)
        S_dim = 2; S_max = [10.0; 1.0];S_min = [0.0; 0.0];
        A_dim = 2; A_max = [pi/2.0;10.0];A_min = [0.0; 0.0];
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [8; 4];
        initializeRBF(S_dim,dimensions,partitions);      
        
        Steps = 1;
        Episodes = 1000;
        evaluateEpisodes = 100;
        
        lambda = 3*10^(-8);
        lb = [-inf(A_dim*RBF.numFeatures,1); [0.01;0.01]];
        ub = [inf(A_dim*RBF.numFeatures,1); [0.2;1.0]];
    case '2_link_1_torque'
        S_dim = 4; S_max = [0.8;4*pi;0.1;4*pi];S_min = [-0.4;-4*pi;-1.5;-4*pi];
        A_dim = 1; A_max = 35.0;A_min = -35.0;
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [2;4;2;4];
        initializeRBF(S_dim,dimensions,partitions);      
        
        Steps = 100*3;
        Episodes = 1;
        evaluateEpisodes = 10;
        
        lambda_min = 0.05;
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 0.5];
    case '2_link_2_torque'
        S_dim = 4; S_max = [0.8;50;0.1;50];S_min = [-0.4;-50;-1.5;-50];
        A_dim = 2; A_max = [70;500];A_min = [-70;-500];
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [2;5;2;5];
        initializeRBF(S_dim,dimensions,partitions);      
        
        Steps = 100*10;
        Episodes = 100;
        evaluateEpisodes = 10;
        
        lambda = 10^(-7);
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01; 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 10.0; 10.0];
end
end

function policy = initGaussianPolicy
global A_dim domain RBF

switch domain
    case 'canon'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = [0.2;1.0];
    case '2_link_1_torque'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = 0.5;
    case '2_link_2_torque'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = [10.0;10.0];
end
end