function [rew] = canonRewardFnc(x,a)
global A_max A_min

a(1,:) = min(a(1,:),A_max(1)-0.001);
a(1,:) = max(a(1,:),A_min(1)+0.001);
a(2,:) = min(a(2,:),A_max(2)-0.001);
a(2,:) = max(a(2,:),A_min(2)+0.001);

m = 1; %weight of the ball (kg)
% Compute the cost (-20 time squared distance)
% time t = 2 * v * sin(alpha) / g
t = (2/9.8)*(a(2)*sin(a(1)));
% Measure the next distance = t * (v*cos(alpha) + (F*t)/(2*m))
distance = t * (a(2)*cos(a(1)) + ((1/(2*m))*x(2)).*t);
% Measure cost
rew = -20*(bsxfun(@minus,distance,x(1))).^2;

% rew = exp(-0.01*(distance-x(1))^2) - 1;
end