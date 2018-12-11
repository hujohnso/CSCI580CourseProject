clear all
close all
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
u0(:,:) = peaks(N-1);
figure
surf(X,Y,u0)

err = immse(A, u0)
        