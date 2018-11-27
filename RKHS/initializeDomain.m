function initializeDomain
global tol_dist lambda_range evel_episodes A_max A_min TotalEpisodes numRuns MaxIter
global S_max S_min A_dim S_dim domain Steps Episodes RBF lambda ALD
switch domain
    case 'canon'
        numRuns=100; MaxIter=100;
        % Cannon Toy task
        % State (d, w):  (distance, wind)
        % Action (alpha, v): (alpha, velocity)
        S_dim = 2; S_max = [10.0; 1.0];S_min = [0.0; 0.0];
        A_dim = 2; A_max = [pi/2.0-0.001;10.0];A_min = [0.0+0.001; 0.0];
        
        % Define RBF
        RBF.numOfCenters = 100;
        %         RBF.sigmas = (S_max-S_min)./(nthroot(RBF.numOfCenters,S_dim));
        RBF.sigmas = [1.1;0.3];
        Steps = 1;
        Episodes = 50;
        TotalEpisodes = 5*Episodes;
        evel_episodes = 20;
        
        % tol_dist = sqrt(sum(RBF.sigmas.^2));
        tol_dist = 0.01;
        lambda = 10^(-7);
        lambda_range = [10^(-3) 10^(3)];
    case 'toy'
        numRuns=50;MaxIter=50;
        
        S_dim = 1; S_max = [4.0];S_min = [-4.0];
        A_dim = 1; A_max = [1.0];A_min = [-1.0];
        
        % Define RBF
        RBF.numOfCenters = 20;
        % RBF.sigmas = (S_max-S_min)./(nthroot(RBF.numOfCenters,S_dim));
%         RBF.sigmas = 0.01; %1
%         RBF.sigmas = 0.05; %2
%         RBF.sigmas = 0.1; %3
        RBF.sigmas = 0.5; %3
        Steps = 20;
        Episodes = 1;
        TotalEpisodes = 5;
        evel_episodes = 20;
        
        lambda = 10^(-1);
        lambda_range = [10^(-3) 10^(3)];
        ALD = 0.001;
    case 'mountain_car'
        numRuns=15;MaxIter=100;
        
        S_dim = 2; S_max = [0.5; 0.07];S_min = [-1.2; -0.07];
        A_dim = 1; A_max = [1.0];A_min = [-1.0];
        
        % Define RBF
        RBF.numOfCenters = 50;
        %         RBF.sigmas = (S_max-S_min)./(nthroot(RBF.numOfCenters,S_dim));
        RBF.sigmas = [0.01;0.001];
        Steps = 100;
        Episodes = 2;
        evel_episodes = 2;
        
        lambda = 10^(-0);
        lambda_range = [10^(-3) 10^(3)];
        ALD = 0.001;
    case '2_link_1_torque'
        numRuns=15;MaxIter=100;
        
        S_dim = 4; S_max = [0.8;4*pi;0.1;4*pi];S_min = [-0.4;-4*pi;-1.5;-4*pi];
        A_dim = 1; A_max = 35.0;A_min = -35.0;
        
        % Define RBF
        dimensions = [1; 2];
        partitions = [2;4;2;4];
        initializeRBF(S_dim,dimensions,partitions);
        
        Steps = 100*3;
        Episodes = 1;
    case '2_link_2_torque'
        numRuns=15;MaxIter=100;
        
        S_dim = 4; S_max = [0.8;4*pi;0.1;4*pi];S_min = [-0.4;-4*pi;-1.5;-4*pi];
        A_dim = 2; A_max = [70;500];A_min = [-70;-500];
        
        % Define RBF
        RBF.numOfCenters = 200;
        %         RBF.sigmas = (S_max-S_min)./(nthroot(RBF.numOfCenters,S_dim));
        RBF.sigmas = [0.3;2.5;0.4;2.5];
        Steps = 100*5;
        Episodes = 100;
        evel_episodes = 10;
        TotalEpisodes = 20;

        lambda = 10^(-7);
        lambda_range = [10^(-2) 10^(2)];
    case 'Swing_Pendulum'
        numRuns=25;MaxIter=100;

        S_dim = 2; S_max = [pi; 4*pi];S_min = [-pi; -4*pi];
        A_dim = 1; A_max = [3.0];A_min = [-3.0];

        % Define RBF
        RBF.numOfCenters = 50;
        RBF.sigmas = (S_max-S_min)./(nthroot(RBF.numOfCenters,S_dim));
        % RBF.sigmas = [0.1;0.2];
        Steps = 100;
        Episodes = 1;
        evel_episodes = 5;
        TotalEpisodes = 5*Episodes;
        
        lambda = 1e-7;
        lambda_range = [10^(-3) 10^(3)];
        ALD = 0.0001;
    case 'drone_navigation'
        numRuns=25;MaxIter=25;
        
        global qrsim_sim state_sim pid
        addpath(['..',filesep,'qrsim',filesep,'sim']);
        addpath(['..',filesep,'qrsim',filesep,'controllers']);
        qrsim_sim = QRSim();
        state_sim = qrsim_sim.init('TaskNavigation');
        pid = VelocityPID(state_sim.DT);
        
        S_dim = 12;
        A_dim = 2;
        A_max = [3.0, 3.0];A_min = [-3.0, -3.0];
        
        % Define RBF
        RBF.numOfCenters = 100;
%         RBF.sigmas = [0.2;0.2;0.2;30.0;30.0;30.0;0.1;0.1;0.1;30.0;30.0;30.0;0.1];
        RBF.sigmas = [0.1;0.1;0.1;0.2;0.2;0.2;0.3;0.3;0.3;0.5;0.5;0.5];
%         RBF.sigmas = [0.05;0.05;0.05;0.1;0.1;0.1;0.15;0.15;0.15;0.25;0.25;0.25];
        Steps = 60;
        Episodes = 1;
        evel_episodes = 5;
        TotalEpisodes = 5*Episodes;
        
        lambda = 10^(-3);
        lambda_range = [10^(-3) 10^(3)];
        ALD = 0.0001;
    case 'ardupilot_navigation'
        numRuns=15;MaxIter=100;
        
        global Quad
        addpath(['..',filesep,'ardupilot']);
        addpath(['..',filesep,'ardupilot',filesep,'utilities']);
%         init_plot;
%         plot_quad_model;
        quad_variables;
        quad_dynamics_nonlinear;
        
        S_dim = 12;
        A_dim = 4;
        A_max = [43.5, 6.25, 6.25, 2.25];A_min = [0, -6.25, -6.25, -2.25];
        
        % Define RBF
        RBF.numOfCenters = 100;
%         RBF.sigmas = [0.1;0.1;0.1;0.2;0.2;0.2;0.1;0.1;0.1;0.2;0.2;0.2];
%         RBF.sigmas = [0.5;0.5;0.5;0.2;0.2;0.2;0.1;0.1;0.1;0.2;0.2;0.2];
        RBF.sigmas = [1.0;1.0;1.0;0.2;0.2;0.2;0.1;0.1;0.1;0.2;0.2;0.2];
        Steps = 100;
        Episodes = 30;
        evel_episodes = 2;
        TotalEpisodes = 5*Episodes;
        
        lambda = 10^(-3);
        lambda_range = [10^(-3) 10^(3)];
        ALD = 0.0001;
end
end