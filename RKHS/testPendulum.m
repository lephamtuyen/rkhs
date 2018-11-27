function testPendulum
clear all; clf;
global A_min A_max S_max S_min

S_max = [pi; 4*pi];S_min = [-pi; -4*pi];
A_max = [3.0];A_min = [-3.0];
length 	= 1.0;

% % states = [inner_velocity outer_velocity inner_angle outer_angle]
states = [-pi; 0.0];
action = 0.3;

h=plot(0,0,'MarkerSize',30,'Marker','.','LineWidth',2);
range=1.1*(length); axis([-range range -range range]); axis square;
set(gca,'nextplot','replacechildren');

while(ishandle(h)==1)
    Xcoord=[0, -length*sin(states(1))];
    Ycoord=[0,length*cos(states(1))];
    set(h,'XData', Xcoord,'YData', Ycoord);
    drawnow;
    
    [states, ~] = swingUpPendulumDrawNextState(states, action);  
    
    delay(0.1);
end
end

function delay(seconds)
tic;
while toc < seconds
end
end