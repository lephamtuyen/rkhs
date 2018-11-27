function [new_state, valid] = droneNavigationDrawNextState(state, action)
global A_max A_min qrsim_sim state_sim pid

action = round(action);
for i=1:size(A_max,1)
    action(i,:) = min(action(i,:),A_max(i));
    action(i,:) = max(action(i,:),A_min(i));
end

% for i = 1:10
    U = pid.computeU(state_sim.platforms{1}.getX(),[action;0],0);
    qrsim_sim.step(U);
%     state = state_sim.platforms{1}.getX();
% end

new_state = state_sim.platforms{1}.getX();new_state=new_state(1:12);
valid = state_sim.platforms{1}.isValid();

end