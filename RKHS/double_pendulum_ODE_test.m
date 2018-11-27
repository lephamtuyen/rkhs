function xdot = double_pendulum_ODE_test(t,x)

m1 = 1.0;           % [kg]      mass of 1st link
m2 = 1.0;           % [kg]      mass of 2nd link
l1 = 1.0;           % [m]       length of 1st pendulum
l2 = 1.0;           % [m]       length of 2nd pendulum
g  = 9.82;          % [m/s^2]   acceleration of gravity
I1 = m1*l1^2/12;     % moment of inertia around pendulum midpoint (inner link)
I2 = m2*l2^2/12;     % moment of inertia around pendulum midpoint (outer link)

u1 = x(5);
u2 = x(6);

xdot=zeros(6,1);

A = [l1^2*(0.25*m1+m2) + I1,      0.5*m2*l1*l2*cos(x(1)-x(3));
    0.5*m2*l1*l2*cos(x(1)-x(3)), l2^2*0.25*m2 + I2          ];
b = [g*l1*sin(x(1))*(0.5*m1+m2) - 0.5*m2*l1*l2*x(4)^2*sin(x(1)-x(3)) + u1;
    0.5*m2*l2*(l1*x(2)^2*sin(x(1)-x(3)) + g*sin(x(3)))  + u2];
x_new = A\b;

xdot(1)=x(2);
xdot(2)=x_new(1);
xdot(3)=x(4);
xdot(4)=x_new(2);
end