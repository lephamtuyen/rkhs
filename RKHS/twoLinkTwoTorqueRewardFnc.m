function rew = twoLinkTwoTorqueRewardFnc(valid, step, x, u)
global Steps
% Normalize phi
phi1 = x(1);
dphi1 = x(2);
phi2 = x(3);
dphi2 = x(4);

% h = (cos(phi1)+cos(phi2) + 2.0)/4.0;
% rew = exp(-0.5*(5*(1-h)^2 + 0.05*x(2)^2 + 0.05*x(4)^2)) - 1;

% h = cos(x(1))+cos(x(3));
% rew = exp(-(2-h)^2 - (x(2)^2 + x(4)^2)) - 1;

% h = cos(x(1))+cos(x(3));
% rew = exp(-(2-h)^2 - 0.1*(5.0-0.01*step)) - 1;

% h = cos(x(1))+cos(x(3)); %good
% rew = exp(-10*(2-h)^2 - (x(2)^2 + x(4)^2) - 0.1*(5.0-0.01*step)) - 1;



% rew = -500*(x(1)^2 + x(3)^2) - u(step)^2;

% h = cos(x(1))+cos(x(3));
% rew = exp(-0.1*(2-h)^2 - (x(2)^2 + x(4)^2)) - 1;

% h = cos(x(1))+cos(x(3));
% rew = exp(-0.1*(2-h)^2) - 1;

% h = (cos(phi1)+cos(phi2) + 2.0)/4.0;
% rew = exp(-2*(1-h)^2 - 0.02*x(2)^2 - 0.06*x(4)^2) - 1;

rew = -1000*x(1)^2 -1000*x(3)^2 - x(2)^2 - x(4)^2;
if (valid == 0)
    rew = rew - 100000*(Steps - step);
end
end