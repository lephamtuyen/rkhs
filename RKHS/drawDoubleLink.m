function drawDoubleLink(episodes, trial)
lengths 	= [1.0, 1.0];

h=plot(0,0,'MarkerSize',30,'Marker','.','LineWidth',2);
range=1.1*(lengths(1)+lengths(2)); axis([-range range -range range]); axis square;
set(gca,'nextplot','replacechildren');
titlename = strcat('Trial ', int2str(trial));
title(titlename);

for i=1:size(episodes.x,2)
    states = episodes.x(:,i);
    Xcoord=[0,-lengths(1)*sin(states(1)),-lengths(1)*sin(states(1))-lengths(2)*sin(states(3))];
    Ycoord=[0,lengths(1)*cos(states(1)),lengths(1)*cos(states(1))+lengths(2)*cos(states(3))];
    set(h,'XData', Xcoord,'YData', Ycoord);
    drawnow;
end
end

function delay(seconds)
tic;
while toc < seconds
end
end