clear;clc;
mu=0.5;omega=1.11;
x=0.5;y=0.5;
x1=2;y1=2;
n=10000;
dt=0.01;
figure
hold on
xlim([-5,5])
ylim([-5,5])
X=zeros([n,1]);
Y=zeros([n,1]);
% for i=1:100
%     dy=(mu*(1-x^2)*y-omega^2*x+0.05*normrnd(0,1))*dt;
%     dx=y*dt;
%
%     x=x+dx;
%     y=y+dy;
%
% end
for i=1:n
    dy=(mu*(1-x^2)*y-omega^2*x+5*normrnd(0,1))*dt;
    dx=y*dt;
    
    x=x+dx;
    y=y+dy;
    
    
    X(i,:)=x;
    Y(i,:)=y;
    
    dy1=(mu*(1-x1^2)*y1-omega^2*x1+5*normrnd(0,1))*dt;
    dx1=y1*dt;
    
    x1=x1+dx1;
    y1=y1+dy1;
    
    
    X1(i,:)=x1;
    Y1(i,:)=y1;
    plot(X,Y)
    plot(X1,Y1)
    drawnow
end

hil=hilbert(X);
atan(real(hil)./imag(hil))
plot(X,imag(hil))