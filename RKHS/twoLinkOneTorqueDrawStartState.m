function x0 = twoLinkOneTorqueDrawStartState
% global my0 S0
% x0 = mvnrnd(my0, S0, 1);
% x0(1) = min(x0(1),0.8);
% x0(1) = max(x0(1),-0.4);
% x0(3) = min(x0(3),1.6);
% x0(3) = max(x0(3),-0.1);

% x0(1) = -0.4+0.8*rand;
% x0(2) = 0.0;
% x0(3) = -0.1+1.6*rand;
% x0(4) = 0.0;

x0(1) = pi;
x0(2) = 0.0;
x0(3) = pi;
x0(4) = 0.0;
end