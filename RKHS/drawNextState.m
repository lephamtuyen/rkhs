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
    case 'drone_navigation'
        [xn, valid] = droneNavigationDrawNextState(x,u);
    case 'ardupilot_navigation'
        [xn, valid] = ardupilotNavigationDrawNextState(x,u);
end
end