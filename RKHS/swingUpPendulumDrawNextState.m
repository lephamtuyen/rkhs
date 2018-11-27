function [new_states, valid] = swingUpPendulumDrawNextState(states, action)
global A_min A_max S_max S_min

new_states = zeros(2,1);

dt = 0.05;
        
valid = 1;

action(1,:) = min(action(1,:),A_max);
action(1,:) = max(action(1,:),A_min);

new_states(1) = states(1) + states(2)*dt + 0.02*randn;
new_states(2) = states(2) + action*dt + 0.04*randn;

new_states(1) = angle_normalize(new_states(1));
new_states(2) = min(new_states(2),S_max(2));
new_states(2) = max(new_states(2),S_min(2));

new_states(1) = new_states(1) + new_states(2)*dt + 0.02*randn;
new_states(2) = new_states(2) + action*dt + 0.04*randn;

new_states(1) = angle_normalize(new_states(1));
new_states(2) = min(new_states(2),S_max(2));
new_states(2) = max(new_states(2),S_min(2));

end
function x_new = angle_normalize(x)
    x_new = (mod((x+pi),(2*pi)) - pi);
end