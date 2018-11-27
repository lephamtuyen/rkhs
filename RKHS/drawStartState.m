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
    case 'drone_navigation'
        x0 = droneNavigationDrawStartState;
    case 'ardupilot_navigation'
        x0 = ardupilotNavigationDrawStartState;
end
end