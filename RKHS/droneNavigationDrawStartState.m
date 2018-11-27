function x0 = droneNavigationDrawStartState
global state_sim qrsim_sim

qrsim_sim.reset();
x0 = state_sim.platforms{1}.getX();
x0 = x0(1:12);
end