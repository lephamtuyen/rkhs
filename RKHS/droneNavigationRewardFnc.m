function [rew] = droneNavigationRewardFnc(x,u)
global qrsim_sim
rew = qrsim_sim.reward();
end