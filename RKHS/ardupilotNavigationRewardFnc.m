function [rew] = ardupilotNavigationRewardFnc(x,u)
global Quad
rew = exp(-sum((Quad.X-1.0)^2+(Quad.Y-1.0)^2+(Quad.Z-1.0)^2));
end