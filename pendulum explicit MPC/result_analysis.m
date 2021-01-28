clear
% close all
% load data
folder = 'data/NN_policy_26-01-2021_19-21-02/';
iter = 15;
% folder = 'data/NN_policy_26-01-2021_17-17-56/';
% iter = 16;
fname = [folder num2str(iter) '/sdpvar.mat'];
load(fname, 'dec_var')
Q1 = dec_var.Q1;
Q2 = dec_var.Q2;
L1 = dec_var.L1;
L2 = dec_var.L2;
L3 = dec_var.L3;
L4 = dec_var.L4;
Yk = dec_var.Yk;
AG = dec_var.AG;
BG = dec_var.BG;
factor = 1;
x1bound = 2.5*factor;
x2bound = 6*factor;
fname = [folder num2str(iter) '/sdpvar.mat'];
load(fname, 'dec_var')
W1 = dec_var.W1;
W2 = dec_var.W2;
W3 = dec_var.W3;

n1 = size(W1,1);
n2 = size(W2,1);
n3 = size(W3,1);
nphi = n1+n2;
nG = size(AG, 1);
b1 = zeros(n1,1);
b2 = zeros(n2,1);
b3 = zeros(n3,1);

% bounds for the inputs to the nonlinearity
xeq = [0.0; 0.0];
v1eq = W1*xeq + b1;
w1eq = tanh(v1eq);
v2eq = W2*w1eq + b2;
w2eq = tanh(v2eq);
v3eq = W3*w2eq + b3; % This is also u_*
% usat = sat(v3) = sat(u)

w0up = [x1bound; x2bound];
w0lb = -w0up;
v1up = [];
v1lb = [];
for i = 1:n1
    v1up_i = W1(i,:)*1/2*(w0up+w0lb) + b1(i) + abs(W1(i,:))*1/2*abs(w0up-w0lb);
    v1lb_i = W1(i,:)*1/2*(w0up+w0lb) + b1(i) - abs(W1(i,:))*1/2*abs(w0up-w0lb);
    v1up = [v1up; v1up_i];
    v1lb = [v1lb; v1lb_i];
end

%
alpha1 = min((tanh(v1up)-tanh(v1eq))./(v1up-v1eq), (tanh(v1eq)-tanh(v1lb))./(v1eq-v1lb));
beta = 1;
w1up = tanh(v1up);
w1lb = tanh(v1lb);

v2up = [];
v2lb = [];
for i = 1:n2
    v2up_i = W2(i,:)*1/2*(w1up+w1lb) + b2(i) + abs(W2(i,:))*1/2*abs(w1up-w1lb);
    v2lb_i = W2(i,:)*1/2*(w1up+w1lb) + b2(i) - abs(W2(i,:))*1/2*abs(w1up-w1lb);
    v2up = [v2up; v2up_i];
    v2lb = [v2lb; v2lb_i];
end
alpha2 = min((tanh(v2up)-tanh(v2eq))./(v2up-v2eq), (tanh(v2eq)-tanh(v2lb))./(v2eq-v2lb));

Alpha = blkdiag(diag(alpha1),diag(alpha2));
Beta = beta*eye(nphi);

%
N = blkdiag(W1,W2,W3);
Nux = N(n1+n2+1:end, 1:nG);
Nuw = N(n1+n2+1:end,nG+1:end);
Nvx = N(1:n1+n2, 1:nG);
Nvw = N(1:n1+n2,nG+1:end);
fNvx = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvx;
fNvw = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);
fNux = Nux + Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvx;
fNuw = Nuw*1/2*(Beta-Alpha)+...
    Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);
fN = [fNux, fNuw;...
      fNvx, fNvw];

%
Q =  blkdiag(Q1,Q2);
L = [L1, L2;...
     L3, L4];

%
N_diff = fN-L*inv(Q);
if true
    figure(1)
    % mesh(N_diff)
    subplot(2,2,1)
    h1 = mesh(fN);
    h1.FaceAlpha = 0.5;
    h1.FaceColor = mycolor('coolblue');
    h1.EdgeColor = mycolor('coolblue');
    h1.EdgeAlpha  = 0.5;
    h1.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(f(N))$','interpreter','latex')
    garyfyFigure
    subplot(2,2,2)
    h2 = mesh(L*inv(Q));
    h2.FaceAlpha = 0.5;
    h2.FaceColor = mycolor('orange');
    h2.EdgeColor = mycolor('orange');
    h2.EdgeAlpha  = 0.5;
    h2.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(LQ^{-1})$','interpreter','latex')
    subplot(2,2,3)
    h3 = mesh(fN*Q);
    h3.FaceAlpha = 0.5;
    h3.FaceColor = mycolor('coolblue');
    h3.EdgeColor = mycolor('coolblue');
    h3.EdgeAlpha  = 0.5;
    h3.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(f(N)*Q)$','interpreter','latex')
    garyfyFigure
    subplot(2,2,4)
    h4 = mesh(L);
    h4.FaceAlpha = 0.5;
    h4.FaceColor = mycolor('orange');
    h4.EdgeColor = mycolor('orange');
    h4.EdgeAlpha  = 0.5;
    h4.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(L)$','interpreter','latex')
    garyfyFigure
end

%%
dec_var.obj

%%
xU = [x1bound; x2bound];
xL = -xU;
num_pt = 50;
x1 = linspace(xL(1), xU(1),num_pt);
x2 = linspace(xL(2),xU(2),num_pt);
ug = zeros(num_pt,num_pt);
for i=1:num_pt
    for j = 1:num_pt
        ug(j,i) = nn_eval(W1,W2,W3,[x1(i); x2(j)]);
    end
end
figure(2)
subplot(1,2,1)
h = mesh(x1,x2,ug);
h.FaceAlpha = 0.5;
h.FaceColor = mycolor('coolblue');
h.EdgeColor = mycolor('coolblue');
h.EdgeAlpha  = 0.4;
hold on
load('exp_data2.csv')
p1 = plot3(exp_data2(:,1),exp_data2(:,2),exp_data2(:,3),'ro','MarkerSize',2);
p1.Color = mycolor('orange');
grid on
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
zlabel('$u$','interpreter','latex')
legend('NN policy', 'expert data', 'interpreter','latex')
title('NN vs. expert data')
garyfyFigure

% ROA
subplot(1,2,2)
p2= plot(exp_data2(:,1),exp_data2(:,2),'ro','MarkerSize',2);
p2.Color = mycolor('orange');
hold on
ROA_NN(fNvx, fNvw, fNux, fNuw, nG, nphi,AG,BG,x1bound,x2bound)
legend('ROA of MPC','$X$', 'ROA of NN' ,'interpreter','latex')
hold on
garyfyFigure

%% 
if true
    figure(3)
    miss_match = [];
    for i = 0:iter
        fname = [folder num2str(i) '/sdpvar.mat'];
        load(fname, 'dec_var')
        miss_match = [miss_match, dec_var.obj(2)];
    end
%     plot(0:iter, miss_match,'-*')
    semilogy(0:iter,miss_match,'-*')
    grid minor
end

%%
function u = nn_eval(W1,W2,W3,x)
W{1} = W1;
W{2} = W2;
W{3} = W3;
z = x;
for i = 1:2
    z = W{i}*z;
    z = tanh(z);
end
u = W{end}*z;
end

function ROA_NN(fNvx, fNvw, fNux, fNuw,nG,nphi,AG,BG,x1bound,x2bound)
% Convex Optimization - compute ROA of the NN controller from NN step
cvx_begin sdp
    cvx_solver mosek
    
    % Variables
    variable Q1(nG,nG) symmetric;
    variable Q2(nphi,nphi) diagonal; 
%     variable L1(nu, nG); % L1 = \tilde{Nux}*Q1;
%     variable L2(nu, nphi); % L2 = \tilde{Nuw}*Q2;
%     variable L3(nphi, nG); % L3 = \tilde{Nvx}*Q1;
%     variable L4(nphi, nphi); % L4 = \tilde{Nvw}*Q2;
    L1 = fNux*Q1; L2 = fNuw*Q2; L3 = fNvx*Q1; L4 = fNvw*Q2;
    Q1 >= 1e-6*eye(nG);
    Q2 >= 1e-6*eye(nphi);
    Q =  blkdiag(Q1,Q2);
    % left upper corner
    LU = Q;
    
    % right lower corner
    RL = Q;
    
    % left lower corber
    LL = [AG*Q1+BG*L1, BG*L2;...
          L3, L4];
    
    % right upper corner
    RU = LL';
    
    % Matrix Inequality
    LMI = [LU, RU;...
           LL, RL];
    LMI >= 0;
    % enforce {x: x'Px<=1} \subset {x: |x1| <= x1bound}
     [1,0]*Q1*[1,0]' <= x1bound^2;
    % enforce {x: x'Px<=1} \subset {x: |x2| <= x2bound}
     [0,1]*Q1*[0,1]' <= x2bound^2;
    % objective function
    obj = -log_det(Q1);
    minimize(obj)
cvx_end

% plot state constraint set 
X = Polyhedron('lb',[-x1bound; -x2bound],'ub',[x1bound; x2bound]);
X.plot('alpha',0.4,'color',mycolor('lightgray'),'linewidth',3,'edgecolor',mycolor('darkgray'))
hold on
% plot ROA
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
domain1 = [-5, 5, -10, 10];
[C,h] = pcontour(V,1,domain1,'r');
h.LineColor = mycolor('coolblue');
h.LineWidth = 3;

grid on;
axis([-x1bound-0.5 x1bound+0.5 -x2bound-1 x2bound+1]);
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
end