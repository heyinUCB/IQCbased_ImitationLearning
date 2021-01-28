function ROA_LQR(AG,BG,K,x1bound,x2bound)
% Convex Optimization - compute ROA of the NN controller from NN step
cvx_begin sdp
    cvx_solver mosek
    nG = size(AG, 1);
    
    % Variables
    variable Q(nG,nG) symmetric;
    Q >= 1e-8*eye(nG);
    
    % Matrix Inequality
    LMI = [Q,           Q*(AG+BG*K')';...
           (AG+BG*K')*Q, Q];
    LMI >= 0;
    
    % enforce {x: x'Px<=1} \subset {x: |x1| <= x1bound}
     [1,0]*Q*[1,0]' <= x1bound^2;
    % enforce {x: x'Px<=1} \subset {x: |x2| <= x2bound}
     [0,1]*Q*[0,1]' <= x2bound^2;
    % objective function
    obj = -log_det(Q);
    minimize(obj)
cvx_end
pvar x1 x2
V = [x1,x2]*inv(Q)*[x1;x2];
domain1 = [-5, 5, -10, 10];
[C,h] = pcontour(V,1,domain1,'--r');
h.LineColor = mycolor('darkgray');
h.LineWidth = 5;
hold on
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
zlabel('$u$','interpreter','latex')
end