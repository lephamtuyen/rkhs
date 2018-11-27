function policy = initPolicy
global A_dim domain RBF S_dim increase_id
increase_id = 0;
policy.id = getPolicyID;
switch domain
    case 'canon'
        policy.type = 'non-parametric';
        policy.params = zeros(A_dim+S_dim,1);
        policy.sigma = [0.15;0.3];
    case 'toy'
        policy.type = 'non-parametric';
        policy.params = zeros(A_dim+S_dim,1);
        policy.sigma = 0.01;
    case 'mountain_car'
        policy.type = 'non-parametric';
        policy.params = zeros(A_dim+S_dim,1);
        policy.sigma = 0.1;
    case '2_link_1_torque'
        policy.w = zeros(RBF.numFeatures, A_dim);
        policy.sigma = 0.5;
    case '2_link_2_torque'
        policy.type = 'non-parametric';
        policy.params = zeros(A_dim+S_dim,1);
        policy.sigma = [5.0;10.0];
    case 'Swing_Pendulum'
        policy.type = 'non-parametric';
        policy.params = zeros(A_dim+S_dim,1);
        policy.sigma = 0.1;
    case 'drone_navigation'
        policy.type = 'non-parametric';
        policy.params = zeros(A_dim+S_dim,1);
%         policy.sigma = [0.5;0.5];
        policy.sigma = [0.3;0.3];
    case 'ardupilot_navigation'
        policy.type = 'non-parametric';
        policy.params = zeros(A_dim+S_dim,1);
        policy.sigma = [1.0;0.1;0.1;0.1];
end
end