function testDoubleLink
clear all; clf;

lengths 	= [1.0, 1.0];

% % states = [inner_velocity outer_velocity inner_angle outer_angle]
% states = [0.0; 0.0; 0.0; 0.0];
% action = [70.0;0.0];

h=plot(0,0,'MarkerSize',30,'Marker','.','LineWidth',2);
range=1.1*(lengths(1)+lengths(2)); axis([-range range -range range]); axis square;
set(gca,'nextplot','replacechildren');

load('good/1028.mat');i=1;
while(ishandle(h)==1)
    states = data.episodes(1).x(:,i);i=i+1;
    Xcoord=[0,-lengths(1)*sin(states(1)),-lengths(1)*sin(states(1))-lengths(2)*sin(states(3))];
    Ycoord=[0,lengths(1)*cos(states(1)),lengths(1)*cos(states(1))+lengths(2)*cos(states(3))];
    set(h,'XData', Xcoord,'YData', Ycoord);
    drawnow;
    
%     [states, ~] = twoLinkTwoTorqueDrawNextState(states, action);  
    
% %     if (states(1) > pi || states(1) < -pi)
%         states(1) = convertAngle(states(1));
% %     end
%     
% %     if (states(3) > pi || states(3) < -pi)
%         states(3) = convertAngle(states(3));
% %     end
%     delay(0.1);
end
end

function delay(seconds)
tic;
while toc < seconds
end
end