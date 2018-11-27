function pg_nac
clear all; %#ok<CLFUN>
close all;
clc;

global lambda gamma domain A_dim RBF lb ub numParams

% domain = 'canon';
% domain = 'mountain_car';
domain = 'toy';
% domain = '2_link_1_torque';
% domain = '2_link_2_torque';
% domain = 'Swing_Pendulum';
% domain = 'drone_navigation';
% domain = 'ardupilot_navigation';

initializeDomain;

% Discounted reward
gamma = 0.99;

% initial update counter
numRuns=50;MaxIter=50;
rew = zeros(numRuns,MaxIter);
numParams = A_dim*RBF.numFeatures+A_dim;
% params = zeros(numParams,MaxIter);

for i=1:numRuns
    fprintf('RUN: %d\n', i);
    
    % Initialize gaussian policy
    policy = initGaussianPolicy;

    e = 1;
    while (e <= MaxIter)
        % Parameters
        theta = [policy.w(:);policy.sigma];
%         params(:,e)=theta;
        
        % Take action, observe reward and next state
        data  = obtainData(policy);
        
% %         draw
%         draw = 1;
%         if (draw==1)
%             drawPendulum(data.episodes(1),e);
% %             drawDoubleLink(data.episodes(1),e);
%         end
        
        % ///////////////////////////// NaturalActorCritic /////////////////////
        dTheta = episodicNaturalActorCritic(policy, data);
        %     dTheta = episodicNaturalActorCriticWithConstantBaseline(policy, data);
%         dTheta = episodicNaturalActorCriticWithVariantBaseline(policy, data);
%         dTheta = naturalActorCriticLSTD(policy, data);
        
        % ///////////////////////////// Line Search ///////////////////////////
        lambda = findTheBestLearningRateUsingLineSearch(dTheta, theta, policy);
%         lambda = 0.1/norm(dTheta);
        % ///////////////////////////// Update policy /////////////////////
        theta_new=kk_proj(theta + lambda * dTheta,ub,lb);
        policy.w = reshape(theta_new(1:A_dim*RBF.numFeatures),[RBF.numFeatures, A_dim]);
        policy.sigma = reshape(theta_new(A_dim*RBF.numFeatures+1:A_dim*RBF.numFeatures+A_dim),[A_dim, 1]);
        % ///////////////////////////// Update policy /////////////////////
        
        rew(i,e) = data.averageReward;
        
        printLog(e, policy, data);
        
        % Increase
        e = e + 1;
    end
end
filename = strcat('data/',domain,'_ac');
save(filename,'rew');
end

function printLog(e, policy, data)
global domain lambda
switch domain
    case 'canon'
        fprintf('step: %d, reward: %f, lambda: %f\n', e, data.averageReward,lambda);
        fprintf('sigma: %f\n', policy.sigma);
    case 'toy'
        fprintf('step: %d, reward: %f, lambda: %f\n', e, data.averageReward,lambda);
        fprintf('x max: %f, x min: %f\n', max(data.episodes(1).x), min(data.episodes(1).x));
        fprintf('action max: %f, action min: %f\n', max(data.episodes(1).u),min(data.episodes(1).u));
        fprintf('sigma: %f\n', policy.sigma);
    case 'mountain_car'
        fprintf('step: %d, reward: %f, steps: %f, lambda: %f\n', e, data.averageReward, mean(data.length),lambda);
        fprintf('x max 1: %f, v max 2: %f\n', max(data.episodes(1).x(1,:)), max(data.episodes(1).x(2,:)));
        fprintf('x min 1: %f, v min 2: %f\n', min(data.episodes(1).x(1,:)), min(data.episodes(1).x(2,:)));
        fprintf('actionmax1: %f, actionmin1: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('sigma: %f\n', policy.sigma);
    case '2_link_1_torque'
        fprintf('step: %d, reward: %f, steps: %f\n', e, lambda, data.averageReward, mean(data.length));
        fprintf('velocity max 1: %f, velocity max 2: %f\n', max(data.episodes(1).x(2,:)), max(data.episodes(1).x(4,:)));
        fprintf('velocity min 1: %f, velocity min 2: %f\n', min(data.episodes(1).x(2,:)), min(data.episodes(1).x(4,:)));
        fprintf('actionmax1: %f, actionmin1: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('actionmax2: %f, actionmin2: %f\n', max(data.episodes(1).u(2,:)),min(data.episodes(1).u(2,:)));
        fprintf('sigma: %f\n', policy.sigma);
    case '2_link_2_torque'
        fprintf('step: %d, reward: %f, steps: %f, lambda: %f\n', e, data.averageReward, mean(data.length),lambda);
        fprintf('velocity max 1: %f, velocity max 2: %f\n', max(data.episodes(1).x(2,:)), max(data.episodes(1).x(4,:)));
        fprintf('velocity min 1: %f, velocity min 2: %f\n', min(data.episodes(1).x(2,:)), min(data.episodes(1).x(4,:)));
        fprintf('actionmax1: %f, actionmin1: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('actionmax2: %f, actionmin2: %f\n', max(data.episodes(1).u(2,:)),min(data.episodes(1).u(2,:)));
        fprintf('sigma: %f\n', policy.sigma);
    case 'Swing_Pendulum'
        fprintf('step: %d, reward: %f, steps: %f, lambda: %f\n', e, data.averageReward, mean(data.length), lambda);
        fprintf('actionmax: %f, actionmin: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('sigma: %f\n', policy.sigma);
end

end

function best_alpha = findTheBestLearningRateUsingLineSearch(dTheta, theta, policy)
global A_dim evel_episodes ub lb RBF lambda_range

best_alpha = 0.0;
best_value = -1000000000000;
numEval = 40;
% for grid = lambda_range(1):(diff(lambda_range)/numEval):lambda_range(2)
for grid = 1:numEval
    alpha_new = 0.01 * (((1.1)^(grid-1))-1) + 0.0000001;
    %     alpha_new = grid;
    
    theta_new=kk_proj(theta + alpha_new * dTheta,ub,lb);
    policy_new = policy;
    policy_new.w = reshape(theta_new(1:A_dim*RBF.numFeatures),[RBF.numFeatures, A_dim]);
    policy_new.sigma = reshape(theta_new(A_dim*RBF.numFeatures+1:A_dim*RBF.numFeatures+A_dim),[A_dim, 1]);

    % Data with new policy
    data_new  = obtainData(policy_new,evel_episodes);
    if (data_new.averageReward > best_value)
        best_value = data_new.averageReward;
        best_alpha = alpha_new;
    end
end
end

function px = kk_proj(x,ub,lb)
px=min(ub,x);
px=max(lb,px);
end

function w = episodicNaturalActorCriticTimeVariantBaseline(policy, data)
global gamma numParams RBF

% OBTAIN GRADIENTS
i=1;
Phi = [];
R = [];
for trial = 1 : size(data.episodes,2)
    first_state = (getRBFFeatures(data.episodes(trial).x(:,1)))';
    first_state = first_state/norm(first_state);
    Phi_i = [zeros(1, numParams), first_state];
    R_phi = 0;
    for step = 1 : size(data.episodes(trial).r,2)
        current_state = data.episodes(trial).x(:,step);
        action = data.episodes(trial).u(:,step);
        next_state = data.episodes(trial).x(:,step+1);
        der = DLogPiDTheta(policy, current_state, action);
        
        Phi_i = Phi_i + gamma^(step-1)*[der' zeros(1,RBF.numFeatures)];
        Phi = [Phi;Phi_i];
        
        R_phi = R_phi + gamma^(step-1)*data.episodes(trial).r(step);
        R = [R;R_phi];
        
        i = i+1;
    end
    
end

w = pinv(Phi'*Phi+eye(numParams + RBF.numFeatures)*1.e-5)*Phi'*R;
w = w(1:numParams);
end

function w = episodicNaturalActorCritic(policy, data)
global gamma numParams

Phi = zeros(numParams + 1, data.numOfEpisodes);
R = zeros(1,data.numOfEpisodes);
for epi = 1 : data.numOfEpisodes
    der = zeros(numParams,data.length(epi));
    for step = 1 : data.length(epi)
        der(:,step) = DLogPiDTheta(policy, data.episodes(epi).x(:,step), data.episodes(epi).u(:,step));
    end

    gammas = gamma.^(0:1:step-1);
    Phi(:,epi) = [(sum(bsxfun(@times,der,gammas),2)); 1];
    R(epi) = sum(bsxfun(@times,data.episodes(epi).r,gammas),2);
end

w = pinv(Phi*Phi'+eye(numParams + 1)*1.e-5)*Phi*R';
w = w(1:numParams);


gradient = zeros(numParams,1);
for epi = 1 : data.numOfEpisodes
    dSumPi = zeros(numParams,1);
    sumR   = 0;
    
    for step = 1 : data.length(epi)
        dSumPi = dSumPi + DLogPiDTheta(policy, data.episodes(epi).x(:,step), data.episodes(epi).u(:,step));
        sumR   = sumR   + gamma^(step-1)*data.episodes(epi).r(step);
    end;
    
    gradient = gradient + dSumPi * sumR;
end

gradient = (1-gamma)*gradient / data.numOfEpisodes;

end

function w = naturalActorCriticLSTD(policy, data)
global gamma numParams RBF

% OBTAIN GRADIENTS
i=1;
Phi = zeros(numParams + RBF.numFeatures, sum(data.length));
Phin = zeros(numParams + RBF.numFeatures, sum(data.length));
R = zeros(1,sum(data.length));
for trial = 1 : data.numOfEpisodes
    nSteps = data.length(trial);
    for step = 1 : nSteps
        current_state = data.episodes(trial).x(:,step);
        action = data.episodes(trial).u(:,step);
        der = DLogPiDTheta(policy, current_state, action);
        
        Phi(:,i) = [der;getRBFFeatures(current_state)];
        
        if(step==nSteps)
            Phin(:,i) = [zeros(numParams,1); zeros(RBF.numFeatures,1)];
        else
            next_state = data.episodes(trial).x(:,step+1);
            Phin(:,i) = [zeros(numParams,1); getRBFFeatures(next_state)];
        end
        R(i) = data.episodes(trial).r(step);
        
        i = i+1;
    end
end

w = pinv(Phi*(Phi-gamma*Phin)'+eye(numParams + RBF.numFeatures)*1.e-5)*Phi*R';
w = w(1:numParams);
end

function grad_nat = episodicNaturalActorCriticWithConstantBaseline(policy, data)
global gamma numParams

sumdlog = zeros(numParams+1, data.numOfEpisodes);
rewgamma = zeros(1,data.numOfEpisodes);
for epi = 1 : data.numOfEpisodes
    der = zeros(numParams,data.length(epi));
    for step = 1 : data.length(epi)
        der(:,step) = DLogPiDTheta(policy, data.episodes(epi).x(:,step), data.episodes(epi).u(:,step));
    end
    
    gammas = gamma.^(0:1:step-1);
    sumdlog(:,epi) = [(sum(bsxfun(@times,der,gammas),2)); 1];
    rewgamma(epi) = sum(bsxfun(@times,data.episodes(epi).r,gammas),2);
end
F = sumdlog*sumdlog'+eye(numParams + 1)*1.e-5;
g = sumdlog*rewgamma';
aR = sum(rewgamma,2);
el = sum(sumdlog,2);

F = F / data.numOfEpisodes;
g = g / data.numOfEpisodes;
el = el / data.numOfEpisodes;
aR = aR / data.numOfEpisodes;

Q = 1 / data.numOfEpisodes * (1 + el' * pinv(data.numOfEpisodes * F - el * el') * el);
b = Q * (aR' - el' * pinv(F) * g);
grad_nat = pinv(F) * (g - el * b);
grad_nat = grad_nat(1:numParams);
end

function grad_nat = episodicNaturalActorCriticWithVariantBaseline(policy, data)
global gamma numParams RBF

phi = zeros(numParams+RBF.numFeatures,max(data.length));
aR = zeros(1,max(data.length));
F = zeros(numParams+RBF.numFeatures,numParams+RBF.numFeatures);
g = zeros(numParams+RBF.numFeatures,1);
for epi = 1 : data.numOfEpisodes
    sumDlogLocal  = zeros(numParams+RBF.numFeatures,1);
    for step = 1:data.length(epi)
        der = [DLogPiDTheta(policy, data.episodes(epi).x(:,step), data.episodes(epi).u(:,step)); getRBFFeatures(data.episodes(epi).x(:,step))];
        sumDlogLocal = sumDlogLocal + der;
        phi(:,step) = phi(:,step) + sumDlogLocal;
        F = F + sumDlogLocal*der';
        g = g + sumDlogLocal*gamma^(step-1)*data.episodes(epi).r(step);
    end
    gammas = gamma.^(0:1:step-1);
    aR(1:step) = aR(1:step) + bsxfun(@times,data.episodes(epi).r,gammas);
end

F = F / data.numOfEpisodes;
g = g / data.numOfEpisodes;
phi = phi / data.numOfEpisodes;
aR = aR / data.numOfEpisodes;

Q = 1 / data.numOfEpisodes * (eye(max(data.length)) + phi' * pinv(data.numOfEpisodes * F - phi * phi') * phi);
b = Q * (aR' - phi' * pinv(F) * g);
grad_nat = pinv(F) * (g - phi * b);
% grad_nat = pinv(F) * g;
grad_nat = grad_nat(1:numParams);
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

function x0 = drawStartState
global domain
switch domain
    case 'canon'
        x0 = canonDrawState;
    case 'toy'
        x0 = toyDrawStartState;
    case 'mountain_car'
        x0 = mountainCarDrawStartState;
    case '2_link_1_torque'
        x0 = twoLinkOneTorqueDrawStartState;
    case '2_link_2_torque'
        x0 = twoLinkTwoTorqueDrawStartState;
    case 'Swing_Pendulum'
        x0 = swingUpPendulumDrawStartState;
end
end

function [xn, valid] = drawNextState(x,u)
global domain
switch domain
    case 'canon'
        xn = canonDrawState;
        valid = 0;
    case 'toy'
        [xn, valid] = toyDrawNextState(x,u);
    case 'mountain_car'
        [xn, valid] = mountainCarDrawNextState(x,u);
    case '2_link_1_torque'
        [xn, valid] = twoLinkOneTorqueDrawNextState(x,u);
    case '2_link_2_torque'
        [xn, valid] = twoLinkTwoTorqueDrawNextState(x,u);
    case 'Swing_Pendulum'
        [xn, valid] = swingUpPendulumDrawNextState(x,u);
end
end

function u = drawAction(policy, x, n)
global domain
switch domain
    case 'canon'
        u = canonDrawAction(policy, x, n);
    case 'toy'
        u = toyDrawAction(policy, x, n);
    case 'mountain_car'
        u = mountainCarDrawAction(policy, x, n);
    case '2_link_1_torque'
        u = twoLinkOneTorqueDrawAction(policy, x, n);
    case '2_link_2_torque'
        u = twoLinkTwoTorqueDrawAction(policy, x, n);
    case 'Swing_Pendulum'
        u = swingUpPendulumDrawAction(policy, x, n);
end
end

function rew = rewardFnc(x, u)
global domain
switch domain
    case 'canon'
        rew = canonRewardFnc(x,u);
    case 'toy'
        rew = toyRewardFnc(x,u);
    case 'mountain_car'
        rew = mountainCarRewardFnc(x,u);
    case '2_link_1_torque'
        rew = twoLinkOneTorqueRewardFnc(valid, step, x, u);
    case '2_link_2_torque'
        rew = twoLinkTwoTorqueRewardFnc(valid, step, x, u);
    case 'Swing_Pendulum'
        rew = swingUpPendulumRewardFnc(x, u);
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
RBF.sigmas = (S_max-S_min)./(RBF.partitions);
RBF.centers = linspace(S_min(1)+RBF.sigmas(1),S_max(1)-RBF.sigmas(1),RBF.partitions(1));
for i=2:RBF.dimens
    temp = linspace(S_min(i)+RBF.sigmas(i),S_max(i)-RBF.sigmas(i),RBF.partitions(i));
    RBF.centers = combvec(RBF.centers,temp);
end
end

function initializeDomain
global lambda_range evel_episodes lambda_min A_max A_min evaluateEpisodes S_max S_min A_dim S_dim domain Steps Episodes RBF lb ub basis_dim lambda rbf_centers rbf_sigma
global qrsim_sim state_sim pid

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
        Episodes = 100;
        evel_episodes = 100; 

        lambda = 2*10^(-2); %NAC
%         lambda = 10^(-3); %NAC constant baseline
%         lambda = 10^(-3); %NAC variant baseline
        lambda_range = [10^-2 10^2];
        lb = [-inf(A_dim*RBF.numFeatures,1); [0.01;0.1]];
        ub = [inf(A_dim*RBF.numFeatures,1); [0.1;5.0]];
    case 'toy'
        S_dim = 1; S_max = [4.0];S_min = [-4.0];
        A_dim = 1; A_max = [1.0];A_min = [-1.0];
        
        % Define RBF
        dimensions = [1];
        partitions = [20];
        initializeRBF(S_dim,dimensions,partitions);  
        
        Steps = 20;
        Episodes = 5;
        evel_episodes = 10;
        
        lambda = 10^(-1);
        lambda_range = [10^(-3) 10^(3)];
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 0.3];
    case 'mountain_car'
        S_dim = 2; S_max = [0.7; 0.07];S_min = [-1.2; -0.07];
        A_dim = 1; A_max = [1.0];A_min = [-1.0];
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [10;5];
        initializeRBF(S_dim,dimensions,partitions);  
%         RBF.sigmas = 2*RBF.sigmas;
        Steps = 100;
        Episodes = 100;
        evel_episodes = 1;
        
        lambda = 10^(2);
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 0.5];
    case '2_link_1_torque'
        S_dim = 4; S_max = [0.8;4*pi;0.1;4*pi];S_min = [-0.4;-4*pi;-1.5;-4*pi];
        A_dim = 1; A_max = 35.0;A_min = -35.0;
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [2;5;2;5];
        initializeRBF(S_dim,dimensions,partitions);      
        
        Steps = 100*3;
        Episodes = 1;
        evel_episodes = 10;
        
        lambda_min = 0.05;
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 0.3];
    case '2_link_2_torque'
        S_dim = 4; S_max = [0.8;4*pi;0.1;4*pi];S_min = [-0.4;-4*pi;-1.5;-4*pi];
        A_dim = 2; A_max = [70;500];A_min = [-70;-500];
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [2;5;2;5];
        initializeRBF(S_dim,dimensions,partitions);      
        RBF.sigmas = [0.3;2.5;0.4;2.5];
        
        Steps = 100*5;
        Episodes = 100;
        evel_episodes = 10;
        
        lambda = 10^(-6);
        lambda_range = [10^(-3) 10^(3)];
        
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01; 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 5.0; 10.0];
     case 'Swing_Pendulum'
        S_dim = 2; S_max = [pi; 8.0];S_min = [-pi; -8.0];
        A_dim = 1; A_max = [3.0];A_min = [-3.0];
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [7;14];
        initializeRBF(S_dim,dimensions,partitions);  
%         RBF.sigmas = RBF.sigmas/2;
        
        Steps = 400;
        Episodes = 100;
        evel_episodes = 2;
        
        lambda = 10^(-1);
        lambda_range = [10^(-3) 10^(3)];
        
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 0.2];
     case 'drone_navigation'
        addpath(['..',filesep,'qrsim',filesep,'sim']);
        addpath(['..',filesep,'qrsim',filesep,'controllers']);
        qrsim_sim = QRSim();
        state_sim = qrsim_sim.init('TaskNavigation');
        pid = VelocityPID(state_sim.DT);
        
        S_dim = 13; 
%         S_max = [pi; 4*pi];S_min = [-pi; -4*pi];
        A_dim = 3; 
        A_max = [16.0, 16.0, 8.0];A_min = [-16.0, -16.0, -8.0];
        
        % Define RBF
        RBF.numOfCenters = 100;
        RBF.sigmas = [0.1;0.1;0.1;0.2;0.2;0.2;0.5;0.5;0.5;0.05;0.05;0.05;1.0];
        Steps = 100;
        Episodes = 1;
        evel_episodes = 5;
        
        lambda = 10^(-1);
        lambda_range = [10^(-3) 10^(3)];
        
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 0.2];
     case 'ardupilot_navigation'
        global Quad
        addpath(['..',filesep,'ardupilot']);
        addpath(['..',filesep,'ardupilot',filesep,'utilities']);
%         init_plot;
%         plot_quad_model;
        quad_variables;
        quad_dynamics_nonlinear;
        
        S_dim = 12; 
        S_max = [pi; 8.0];
        S_min = [-pi; -8.0];
        A_dim = 4;
        A_max = [43.5, 6.25, 6.25, 2.25];A_min = [0, -6.25, -6.25, -2.25];
        
        % Define RBF
        dimensions = [1;2;3;4;5;6;7;8;9;10;11;12];
        partitions = [2;2;2;2;2;2;2;2;2;2;2;2];
        initializeRBF(S_dim,dimensions,partitions);  
%         RBF.sigmas = RBF.sigmas/2;
        
        Steps = 100;
        Episodes = 100;
        evel_episodes = 2;
        
        lambda = 10^(-1);
        lambda_range = [10^(-3) 10^(3)];
        
        lb = [-inf(A_dim*RBF.numFeatures,1); 0.1; 0.01; 0.01; 0.01];
        ub = [inf(A_dim*RBF.numFeatures,1); 1.0; 0.5; 0.5; 0.5];
end
end

function policy = initGaussianPolicy
global A_dim domain RBF

switch domain
    case 'canon'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = [0.1;5.0];
    case 'toy'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = 0.15;
    case 'mountain_car'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = 0.5;
    case '2_link_1_torque'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = 0.5;
    case '2_link_2_torque'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = [5.0;10.0];
    case 'Swing_Pendulum'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = 0.2;
    case 'ardupilot_navigation'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = [1.0;0.1;0.1;0.1];
end
end