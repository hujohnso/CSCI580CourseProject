lP = 90 * 90;
loa = 91 * 91;
lx = 1;
ly = 1;
nx = 90;
ny = 90;
x = linspace(0,lx,nx+1); hx = lx/nx;
y = linspace(0,ly,ny+1); hy = ly/ny;
load('../Q.dat')
load('../P.dat')
load('../Q_True.dat')
load('../P_True.dat')

%If you replace this next line you can see that this is ploting properly
%NavierStokes = NavierStokesTrue
% NavierStokes = NavierStokes';
% P = reshape(NavierStokes(1:lP), [90 90]);
% Q = reshape(NavierStokes(lP + 1: loa + lP), [91 91]);
% VFFA = reshape(NavierStokes(loa + lP + 1: loa * 2 + lP ), [91 91]);
% VFSA = reshape(NavierStokes(loa * 2 + lP  + 1: loa * 3 + lP), [91 91]);
N = 90; % Number of grid in x,y-direction
L = 1; % Domain size

P = reshape(P, [90 90]);
Q = reshape(Q, [91 91]);
P_True = reshape(P_True, [90 90]);
Q_True = reshape(Q_True, [91 91]);

bottom = min(min(min(P)),min(min(P_True)));
top  = max(max(max(P)),max(max(P_True)));

figure
clf, contourf(x(1 : 90),y(1:90),P_True',20,'w-'), hold on
contour(x,y,Q_True',5,'k-');
caxis manual
caxis([bottom top]);
colormap hsv
cmap = colormap;
hold off, axis equal, axis([0 lx 0 ly])
title('Original Solution')
drawnow

figure
clf, contourf(x(1 : 90),y(1:90),P',20,'w-'), hold on
contour(x,y,Q',5,'k-');
caxis manual
caxis([bottom top]);
colormap hsv
hold off, axis equal, axis([0 lx 0 ly])
title('NNFDA Solution')
drawnow



err_P = immse(P, P_True)
err_Q = immse(Q, Q_True)


% function B = avg(A,k)
% if nargin<2, k = 1; end
% if size(A,1)==1, A = A'; end
% if k<2, B = (A(2:end,:)+A(1:end-1,:))/2; else, B = avg(A,k-1); end
% if size(A,2)==1, B = B'; end