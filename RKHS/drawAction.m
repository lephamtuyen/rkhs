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
    case 'drone_navigation'
        u = droneNavigationDrawAction(policy, x, n);
    case 'ardupilot_navigation'
        u = ardupilotNavigationDrawAction(policy, x, n);
end
end