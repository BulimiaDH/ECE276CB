A = [0.75, -1; 1, 0.75];
B = [1, 0.5; 0.5, 0.5];
xv = -1:0.01:1;
yv = 1:-0.01:-1;
n = 2/(0.01) + 1;
[X,Y] = meshgrid(xv,yv);
xlist = [X(:),Y(:)]';
V =@(x) 1/2 * x' * x;
V2 =@(x) min([V(x) + V(A*x),V(x) + V(B*x)]);
V1 =@(x) min(V(x) + V2(A*x),V(x) + V2(B*x));
V0 = @(x) min(V(x) + V1(A*x),V(x) + V1(B*x));
pi2 = zeros(n,n);
pi1 = zeros(n,n);
pi0 = zeros(n,n);
V0m = zeros(n,n);
for i = 1:n
    for j = 1:n
        x = xlist(:,(i-1)*n + j);
        [M,I] = min([V(x) + V(A*x),V(x) + V(B*x)]);
        pi2(j,i) = I-1;
        [M,I] = min([V(x) + V2(A*x),V(x) + V2(B*x)]);
        pi1(j,i) = I-1;
        [M,I] = min([V(x) + V1(A*x),V(x) + V1(B*x)]);
        pi0(j,i) = I-1;
        V0m(j,i) = V0(x);
    end
end
xa = -1:0.01:1;
ya = 1:-0.01:-1;
figure;
imagesc(xa,ya,pi2);
colormap(gray(255))
title('policy2')
xlabel('x')
ylabel('y')
saveas(gcf,'p2.png');

figure;
imagesc(xa,ya,pi1);
colormap(gray(255))
title('policy1')
xlabel('x')
ylabel('y')
saveas(gcf,'p1.png');

figure;
imagesc(xa,ya,pi0);
colormap(gray(255))
title('policy0')
xlabel('x')
ylabel('y')
saveas(gcf,'p0.png');

figure;
imagesc(xa,ya,V0m);
title('V0')
xlabel('x')
ylabel('y')
colorbar
saveas(gcf,'V0.png');

% M1 = A' * A ;
% M2 = B' * B ;
% 
% % syms x y
% % X = [x;y];
% % circle = X'* (M1- M2)* X == 0 ;
% % ezplot(circle, [-1,1,-1,1])
% 
% M11 = A' * A' * A * A;
% M22 = B' * B' * B * B;
% M12 = A' * B' * B * A;
% M21 = B' * A' * A * B;
% 
% % syms x y
% % X = [x;y];
% % f1 = feval(symengine,'min',X'* (M11 + M1)* X,X'* (M12 + M1)* X);
% % f2 = feval(symengine,'min',X'* (M22 + M2)* X,X'* (M21 + M2)* X);
% % circle = f1 == f2 ;
% % ezplot(circle, [-1,1,-1,1])
% 
% syms x y
% X = [x;y];
% X1 = A * [x;y];
% X2 = B * [x;y];
% f11 = feval(symengine,'min',X1'* (M11 + M1)* X1,X1'* (M12 + M1)* X1);
% f21 = feval(symengine,'min',X1'* (M22 + M2)* X1,X1'* (M21 + M2)* X1);
% 
% f12 = feval(symengine,'min',X2'* (M11 + M1)* X2,X2'* (M12 + M1)* X2);
% f22 = feval(symengine,'min',X2'* (M22 + M2)* X2,X2'* (M21 + M2)* X2);
% fA = feval(symengine,'min',f11,f21);
% fB = feval(symengine,'min',f12,f22);
% 
% fsurf(fA, [-1,1,-1,1])