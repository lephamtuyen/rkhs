% function testDrone
% 
% clear all
% close all
% clc
% global qrsim_sim state_sim pid A_max A_min
% A_max = [3.0, 3.0];A_min = [-3.0, -3.0];
% 
% addpath(['..',filesep,'qrsim',filesep,'sim']);
% addpath(['..',filesep,'qrsim',filesep,'controllers']);
% % create simulator object
% qrsim_sim = QRSim();
% 
% % load task parameters and do housekeeping
% state_sim = qrsim_sim.init('TaskNavigation');
% 
% % number of steps we run the simulation for
% load('data/drone_navigation_trajectory_2_10.mat');
% vt = evaluate.episodes(4).u;
% pid = VelocityPID(state_sim.DT);
% 
% tstart = tic;
% state = state_sim.platforms{1}.getX();
% i=1;
% rew = 0;
% while (i<=60),
%     tloop=tic;
%     % one should alway make sure that the uav is valid
%     % i.e. no collision or out of area event happened
%     [state, valid] = droneNavigationDrawNextState(state_sim.platforms{1}.getX(), vt(:,i));
%     rew = rew + 0.99^(i-1) * qrsim_sim.reward();
%     if(~valid)
%         break;
%     end
%     i=i+1;
%     % wait so to run in real time
%     wait = max(0,state_sim.task.dt-toc(tloop));
%     pause(wait);
% end
% state
% 
% rew
% % get reward
% % qrsim.reward();
% 
% elapsed = toc(tstart)
% 
% end

function testDrone

clear all
close all
clc
global qrsim state pid

addpath(['..',filesep,'qrsim',filesep,'sim']);
addpath(['..',filesep,'qrsim',filesep,'controllers']);

% number of steps we run the simulation for
load('data/xxx/drone_navigation_trajectory_2_9.mat');
wps = evaluate.episodes(4).x(1:3,:);
% create simulator object
qrsim = QRSim();

% load task parameters and do housekeeping
state = qrsim.init('TaskNavigation');

% number of steps we run the simulation for
N = 30000;
% the PID controller
pid = WaypointPID(state.DT);

wpidx = 1;
for i=1:60,
    % one should alway make sure that the uav is valid
    % i.e. no collision or out of area event happened
    if(state.platforms{1}.isValid())
        tloop = tic;
%         ex = state.platforms{1}.getX();
%         if(norm(ex(1:3)-wps(:,wpidx))<0.3)
%             wpidx = wpidx+1;
%         end
%         
%         if(wpidx>size(wps,2))
%             break;
%         end
        for j=1:10
            % compute controls
            U = pid.computeU(state.platforms{1}.getX(),wps(:,i),0);
    %         wpidx
            % step simulator
            qrsim.step(U);  
        end
    end
    % wait so to run in real time
    wait = max(0,state.task.dt-toc(tloop));
    pause(wait);
    
end

end