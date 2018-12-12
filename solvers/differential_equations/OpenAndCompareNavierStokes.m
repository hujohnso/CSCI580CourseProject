
lP = 90 * 90;
loa = 91 * 91;
lx = 1;
ly = 1;
nx = 90;
ny = 90;
x = linspace(0,lx,nx+1); hx = lx/nx;
y = linspace(0,ly,ny+1); hy = ly/ny;
load('../NavierStokes.dat')
load('../NavierStokesTrue.dat')
%If you replace this next line you can see that this is ploting properly
%NavierStokes = NavierStokesTrue
NavierStokes = NavierStokes';
P = reshape(NavierStokes(1:lP), [90 90]);
Q = reshape(NavierStokes(lP + 1: loa + lP), [91 91]);
VFFA = reshape(NavierStokes(loa + lP + 1: loa * 2 + lP ), [91 91]);
VFSA = reshape(NavierStokes(loa * 2 + lP  + 1: loa * 3 + lP), [91 91]);
N = 90; % Number of grid in x,y-direction
L = 1; % Domain size

%  clf, contourf(avg(x),avg(y),P',20,'w-'), hold on
clf, contourf(x(1 : 90),y(1:90),P',20,'w-'), hold on
contour(x,y,Q',20,'k-');
quiver(x,y,VFFA,VFSA,.4,'k-')
hold off, axis equal, axis([0 lx 0 ly])
drawnow
err = immse(NavierStokesTrue, NavierStokes)

% function B = avg(A,k)
% if nargin<2, k = 1; end
% if size(A,1)==1, A = A'; end
% if k<2, B = (A(2:end,:)+A(1:end-1,:))/2; else, B = avg(A,k-1); end
% if size(A,2)==1, B = B'; end