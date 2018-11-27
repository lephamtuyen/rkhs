function x0 = ardupilotNavigationDrawStartState
global Quad
quad_variables;
quad_dynamics_nonlinear;  

x0 = [Quad.X; Quad.Y; Quad.Z; ...
    Quad.X_dot; Quad.Y_dot; Quad.Z_dot; ...
    Quad.phi; Quad.theta; Quad.psi; ...
    Quad.p; Quad.q; Quad.r];

Quad.init = 1;
end