function [new_states, valid] = mountainCarDrawNextState(states, action)
global A_max A_min S_min S_max
% new_states = zeros(2,1);
% 
action(1,:) = min(round(action(1,:)),A_max);
action(1,:) = max(round(action(1,:)),A_min);
% 
% % x
% new_states(1) = states(1) + states(2);
% new_states(1) = min(new_states(1),S_max(1));
% new_states(1) = max(new_states(1),S_min(1));
% 
% % v
% new_states(2) = states(2) + 0.001*action - 0.0025*cos(3*states(1));
% new_states(2) = min(new_states(2),S_max(2));
% new_states(2) = max(new_states(2),S_min(2));
% 
% if (new_states(1)==S_min(1) && new_states(2) < 0)
%     new_states(2) = 0;
% end

pos = states(1);
vel = states(2);
% new_pos = pos + vel;
% new_vel = vel +  action*0.001 - 0.0025*cos(3*pos);
% 
% if ( new_pos < S_min(1) || new_pos > S_max(1) )
%     new_vel = 0;
% end
% 
% new_vel = min(max(new_vel, S_min(2)), S_max(2));
% new_pos = min(max(new_pos, S_min(1)), S_max(1));
% 
% vel = new_vel;
% pos = new_pos;
new_pos = pos + vel + 0.01*randn;
new_vel = vel +  action*0.001 - 0.0025*cos(3*pos) + 0.001*randn;

if ( new_pos < S_min(1) || new_pos > S_max(1) )
    new_vel = 0;
end
new_vel = min(max(new_vel, S_min(2)), S_max(2));
new_pos = min(max(new_pos, S_min(1)), S_max(1));

new_states = [new_pos;new_vel];

valid = 1;
end