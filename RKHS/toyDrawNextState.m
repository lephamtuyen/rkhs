function [new_state, valid] = toyDrawNextState(state, action)
global A_max A_min

action(1,:) = min(action(1,:),A_max);
action(1,:) = max(action(1,:),A_min);

% if (state < S_min || state > S_max)
%     new_state = 0.0;
% else
    new_state = state + action + 0.01*randn;
% end

% if (new_state < S_min || new_state > S_max)
%     new_state = 0;
% end

% new_state = min(new_state,S_max);
% new_state = max(new_state,S_min);
valid = 1;
end