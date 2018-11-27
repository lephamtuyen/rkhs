function new_angle = convertAngle(angle)
new_angle = angle;
% Convert to [-pi => pi]
if(new_angle > 2*pi)
    new_angle = mod(new_angle,2*pi);
end

% Convert to [-pi => pi]
if(new_angle < -2*pi)
    new_angle = mod(new_angle,-2*pi);
end

% Convert to [-pi => pi]
if(new_angle > pi)
    new_angle = -(2*pi - new_angle);
end

if(new_angle < -pi)
    new_angle = (2*pi + new_angle);
end
end