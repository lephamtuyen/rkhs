 function [new_states, valid] = twoLinkTwoTorqueDrawNextState(states, action)

input = [states; action];
sol=ode45(@double_pendulum_ODE_test,[0 0.01], input);
t = linspace(0,0.01,2);
y=deval(sol,t);

new_states = zeros(4,1);
new_states(1) = convertAngle(y(1,2)); %phi1
new_states(2) = y(2,2); %dphi1
new_states(3) = convertAngle(y(3,2)); %phi2
new_states(4) = y(4,2); %dphi2

phi1 = new_states(1); %phi1
phi2 = new_states(3); %phi2

if (phi1 < -0.4 || phi1 > 0.8 || phi2 < -1.5 || phi2 > 0.1)
    valid = 0;
else
    valid = 1;
end
end