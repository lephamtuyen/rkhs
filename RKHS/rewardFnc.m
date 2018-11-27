function rew = rewardFnc(x, a)
global domain
switch domain
    case 'canon'
        rew = canonRewardFnc(x,a);
    case 'toy'
        rew = toyRewardFnc(x,a);
    case 'mountain_car'
        rew = mountainCarRewardFnc(x,a);
    case '2_link_1_torque'
        rew = twoLinkOneTorqueRewardFnc(step, x, a);
    case '2_link_2_torque'
        rew = twoLinkTwoTorqueRewardFnc(step, x, a);
    case 'Swing_Pendulum'
        rew = swingUpPendulumRewardFnc(x, a);
    case 'drone_navigation'
        rew = droneNavigationRewardFnc(x, a);
    case 'ardupilot_navigation'
        rew = ardupilotNavigationRewardFnc(x, a);
end
end