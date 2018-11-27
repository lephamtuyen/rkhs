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
        fprintf('step: %d, lambda: %f, reward: %f steps: %f\n', e, lambda, data.averageReward,mean(data.length));
        fprintf('x max: %f, x min: %f\n', max(data.episodes(1).x(1,:)), min(data.episodes(1).x(1,:)));
        fprintf('action max: %f, action min: %f\n', max(data.episodes(1).u),min(data.episodes(1).u));
        fprintf('sigma: %f\n', policy.sigma);
    case '2_link_1_torque'
        fprintf('step: %d, reward: %f, steps: %f\n', e, lambda, data.averageReward, mean(data.length));
        fprintf('velocity max 1: %f, velocity max 2: %f\n', max(data.episodes(1).x(2,:)), max(data.episodes(1).x(4,:)));
        fprintf('velocity min 1: %f, velocity min 2: %f\n', min(data.episodes(1).x(2,:)), min(data.episodes(1).x(4,:)));
        fprintf('actionmax1: %f, actionmin1: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('actionmax2: %f, actionmin2: %f\n', max(data.episodes(1).u(2,:)),min(data.episodes(1).u(2,:)));
        fprintf('sigma: %f\n', policy.sigma);
    case '2_link_2_torque'
        fprintf('step: %d, reward: %f, steps: %f, lambda: %f\n', e, data.averageReward, mean(data.length), lambda);
        fprintf('velocity max 1: %f, velocity max 2: %f\n', max(data.episodes(1).x(2,:)), max(data.episodes(1).x(4,:)));
        fprintf('velocity min 1: %f, velocity min 2: %f\n', min(data.episodes(1).x(2,:)), min(data.episodes(1).x(4,:)));
        fprintf('actionmax1: %f, actionmin1: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('actionmax2: %f, actionmin2: %f\n', max(data.episodes(1).u(2,:)),min(data.episodes(1).u(2,:)));
        fprintf('sigma: %f\n', policy.sigma);
    case 'Swing_Pendulum'
        fprintf('step: %d, reward: %f, steps: %f, lambda: %f\n', e, data.averageReward, mean(data.length), lambda);
        fprintf('actionmax: %f, actionmin: %f\n', max(data.episodes(1).u(1,:)),min(data.episodes(1).u(1,:)));
        fprintf('sigma: %f\n', policy.sigma);
    case 'drone_navigation'
        fprintf('step: %d, reward: %f, steps: %f, lambda: %f\n', e, data.averageReward, mean(data.length), lambda);
    case 'ardupilot_navigation'
        fprintf('step: %d, reward: %f, steps: %f, lambda: %f\n', e, data.averageReward, mean(data.length), lambda);
end

end