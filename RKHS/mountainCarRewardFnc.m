function [rew] = mountainCarRewardFnc(x,u)

rew = exp(-8*(x(1)-0.6)^2);
% rew = -1;

end