function [rew] = swingUpPendulumRewardFnc(x,u)
% h=cos(x(1));
% rew = exp(-0.5*(1-h)^2);
rew = exp(-0.5*(x(1))^2);
% rew = -((angle_normalize(x(1)))^2 + .1*x(2)^2 + .001*(u^2));
end

function x_new = angle_normalize(x)
    x_new = (mod((x+pi),(2*pi)) - pi);
end