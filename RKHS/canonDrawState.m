function x = canonDrawState
% % x = zeros(2,1);
% % Random distance is in range [0~10]
% x(1) = 10*rand;
% % Random horizonal wind force in range [0~1]
% x(2) = rand;

% x(1) = 10;
% x(2) = 0;

states = combvec(linspace(0,10,10),linspace(0,1,2));
x = states(:,randi([1 size(states,2)],1,1));
end