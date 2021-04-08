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
x1bound = 2;
x2bound = 5;
x3bound = 1;
x4bound = 5;
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
xeq = zeros(nG,1);
v1eq = W1*xeq + b1;
w1eq = tanh(v1eq);
v2eq = W2*w1eq + b2;
w2eq = tanh(v2eq);
v3eq = W3*w2eq + b3; % This is also u_*
% usat = sat(v3) = sat(u)

w0up = [x1bound; x2bound; x3bound; x4bound];
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
% norm(fN-L*inv(Q),'f')

%% ROA
figure(1)
[Vval,x] = ROA_NN(fNvx, fNvw, fNux, fNuw, nG, nphi,AG,BG,x1bound,x2bound,x3bound,x4bound,ROAcolor);
hold on
% garyfyFigure

%% simulation
xNN = [x0in];
xNNcurr = x0in;
uNN = [];
timer = tic;
for i = 1:num_sim
    uNNcurr = nn_eval(W1,W2,W3,xNNcurr);
    xNNnext = AG*xNNcurr + BG*uNNcurr;
    xNN = [xNN, xNNnext];
    uNN = [uNN, uNNcurr];
    xNNcurr = xNNnext;
end
toc(timer)
dt = 0.02;
figure(2)
subplot(5,1,1)
plot(dt*(0:num_sim), xNN(1,:),simcolor)
hold on
ylabel('$e$','interpreter','latex')

subplot(5,1,2)
plot(dt*(0:num_sim), xNN(2,:),simcolor)
hold on
ylabel('$\dot{e}$','interpreter','latex')

subplot(5,1,3)
plot(dt*(0:num_sim), xNN(3,:),simcolor)
hold on
ylabel('$e_\theta$','interpreter','latex')

subplot(5,1,4)
plot(dt*(0:num_sim), xNN(4,:),simcolor)
hold on
ylabel('$\dot{e}_\theta$','interpreter','latex')

subplot(5,1,5)
plot(dt*(0:num_sim-1), uNN,simcolor)
hold on
ylabel('$u$','interpreter','latex')

%%
function [Vval, x] = ROA_NN(fNvx, fNvw, fNux, fNuw,nG,nphi,AG,BG,x1bound,x2bound,x3bound,x4bound,ROAcolor)
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
    Q1 >= 1e-8*eye(nG);
    Q2 >= 1e-8*eye(nphi);
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
     [1,0,0,0]*Q1*[1,0,0,0]' <= x1bound^2;
    % enforce {x: x'Px<=1} \subset {x: |x2| <= x2bound}
     [0,1,0,0]*Q1*[0,1,0,0]' <= x2bound^2;
    % enforce {x: x'Px<=1} \subset {x: |x3| <= x1bound}
     [0,0,1,0]*Q1*[0,0,1,0]' <= x3bound^2;
    % enforce {x: x'Px<=1} \subset {x: |x4| <= x2bound}
     [0,0,0,1]*Q1*[0,0,0,1]' <= x4bound^2;
    % objective function
    obj = -log_det(Q1);
    minimize(obj)
cvx_end

% plot ROA
pvar x1 x2 x3 x4
x = [x1;x2;x3;x4];
P = inv(Q1);
Vval = x'*P*x;
plotaxes1 = [-x1bound x1bound -x2bound x2bound];
plotaxes2 = [-x3bound x3bound -x4bound x4bound];
% alpha_use = alpha_list(end);
gamma_use = 1;

%
subplot(1,2,1)
x3x4 = [0;0];
V12 = subs(Vval,[x3;x4],x3x4);
[C1,h1] = pcontour(V12, gamma_use, plotaxes1,'r',[500, 500]);
h1.LineColor = ROAcolor;
h1.LineWidth = 4;
hold on

r = rectangle('Position',[-x1bound, -x2bound, 2*x1bound, 2*x2bound]);
r.EdgeColor = mycolor('maroon');
r.LineWidth = 3;
xlabel('$e$','interpreter','latex')
ylabel('$\dot{e}$','interpreter','latex')
%
subplot(1,2,2)
x1x2 = [0;0];
V34 = subs(Vval,[x1;x2],x1x2);
[C2,h2] = pcontour(V34, gamma_use, plotaxes2,'r',[500, 500]);
h2.LineColor = ROAcolor;
h2.LineWidth = 4;
hold on

r = rectangle('Position',[-x3bound, -x4bound, 2*x3bound, 2*x4bound]);
r.EdgeColor = mycolor('maroon');
r.LineWidth = 3;
xlabel('$e_\theta$','interpreter','latex')
ylabel('$\dot{e}_\theta$','interpreter','latex')

end