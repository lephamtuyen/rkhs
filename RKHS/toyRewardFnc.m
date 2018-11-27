function [rew] = toyRewardFnc(x,u)

rew = exp(-abs(x-3));
end