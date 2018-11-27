function [new_state, valid] = ardupilotNavigationDrawNextState(state, action)
global A_max A_min Quad

Quad.U1 = min(action(1),A_max(1));
Quad.U1 = max(Quad.U1,A_min(1));

Quad.U2 = min(action(2),A_max(2));
Quad.U2 = max(Quad.U2,A_min(2));

Quad.U3 = min(action(3),A_max(3));
Quad.U3 = max(Quad.U3,A_min(3));

Quad.U4 = min(action(4),A_max(4));
Quad.U4 = max(Quad.U4,A_min(4));

% Calculate Desired Motor Speeds
quad_motor_speed;

% Update Position With The Equations of Motion
quad_dynamics_nonlinear;

new_state = [Quad.X; Quad.Y; Quad.Z; ...
    Quad.X_dot; Quad.Y_dot; Quad.Z_dot; ...
    Quad.phi; Quad.theta; Quad.psi; ...
    Quad.p; Quad.q; Quad.r];

valid = 1;

end