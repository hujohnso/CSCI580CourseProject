%clear all
%close all
num = 1500;
load('../output.dat')
A = reshape(output, [49 49]);
A = A';
N = 50; % Number of grid in x,y-direction
L = 4*pi; % Domain size
% Grid point
x = linspace(0,L,N);
y = linspace(0,L,N);
% Make it staggered.
x = (x(1:end-1)+x(2:end))/2;
y = (y(1:end-1)+y(2:end))/2;
[X,Y] = meshgrid(x,y);
figure
surf(X,Y,A);
title('NNFDA Approximation at t = 0', 'FontSize', 30)
u0(:,:) = peaks(N-1);
xlim([0 15])
ylim([0 15])
zlim([-10 10])
figure
surf(X,Y,squeeze(u(num,:,:)))
title('ode15s Result at t = 0', 'FontSize',30)
xlim([0 15])
ylim([0 15])
zlim([-10 10])

err = immse(A, squeeze(u(num,:,:)))

figure
surf(X,Y, (abs(squeeze(u(num,:,:)) - A)))
title('Difference Graph at t = 0', 'FontSize',30)
view(0, 90)
colormap hsv
colorbar
        