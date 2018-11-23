clc 
clear
myu=0.000012;
den=1.29;
x0=0;
y0=0;
LX=1;
LY=1;
M=30;
N=30;
dx=(x0+LX)/M;
dy=(y0+LY)/N;
[x,y]=meshgrid(x0:dx:LX,y0:dy:LY);
plot(x,y,'*r');hold on;grid on
[xx,yy]=meshgrid(0.1:0.1:1.1,0.1:0.1:1.1);
plot(xx,yy,'*k');

for i=1:M;
    for j=1:N;
        u(i,j)=randn(1,1);
        v(i,j)=randn(1,1);
        uu(i,j)=3;
        vv(i,j)=1;
        U(i,j)=u(i,j)+uu(i,j);
        V(i,j)=v(i,j)+vv(i,j);
    end
end

for i=1:M;
    for j=1:N;
        x1(i,j)=i*dx;
        y1(i,j)=j*dy;
    end
end

for i=1:M;
    for j=1:N;
        dudy(i,j)=(u(i+1)-u(i))/dy;
        dvdx(i,j)=(v(i+1)-v(i))/dx;
        duudy(i,j)=(uu(i+1)-uu(i))/dy;
        dvvdx(i,j)=(vv(i+1)-vv(i))/dx;
        s(i,j)=0.5*(dudy(i,j)+dvdx(i,j));
        ss(i,j)=0.5*(duudy(i,j)+dvvdx(i,j));
    end
end

for i=1:M;
    for j=1:N;
        if (i==j)
            c=1;
        elseif (i~=j)
            c=0;
        end
        px(i,j)=0.1*c*randn(1,1);
        py(i,j)=0.1*c*randn(1,1);
        ppx(i,j)=0.3*c;
        ppy(i,j)=0.1*c;
        Px(i,j)=px(i,j)+ppx(i,j);
        Py(i,j)=py(i,j)+ppy(i,j);
        P(i,j)=Px(i,j)+Py(i,j);
    end
end

for i=1:M;
    for j=1:N;
        Mstress(i,j)=-(ppx(i,j)/dx+ppy(i,j)/dy)+2*myu*ss(i,j);
        stressf(i,j)=-(px(i,j)/dx+py(i,j)/dy)+2*myu*s(i,j);
        for a=1:10;
        uuu(a)=randn(1,1);
        vvv(a)=randn(1,1);
        end
        um(i,j)=mean(uuu);
        vm(i,j)=mean(vvv);
        rt=um(i,j)*vm(i,j);
        rtt=mean(rt);
        rrtt=den*rtt;
        rrrtt=mean(rrtt);
        ReynoldsStress(i,j)=den*rrrtt;
    end
end

[C,h] = contour(x1,y1,P);
set(1, 'units', 'centimeters', 'pos', [0 0 120.5 100])
colormap summer

quiver(x1,y1,U,V)