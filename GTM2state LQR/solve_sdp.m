% Region of attraction analysis on the pendulum system with NN controller
function dec_var = solve_sdp(param)

AG = param.AG;
BG = param.BG;
nG = size(AG, 1);
nu = size(BG, 2);

%% load weights and biases of the NN controller
W1 = param.W1;
W2 = param.W2;
W3 = param.W3;
n1 = size(W1,1);
n2 = size(W2,1);
n3 = size(W3,1);
nphi = n1+n2;

b1 = zeros(n1,1);
b2 = zeros(n2,1);
b3 = zeros(n3,1);

%% bounds for the inputs to the nonlinearity
xeq = zeros(nG, 1);
v1eq = W1*xeq + b1;
w1eq = tanh(v1eq);
v2eq = W2*w1eq + b2;
w2eq = tanh(v2eq);
v3eq = W3*w2eq + b3; % This is also u_*
% usat = sat(v3) = sat(u)

% -x1bound <= x1 <= x1bound
x1bound = param.x1bound;
% -x2bound <= x2 <= x2bound
x2bound = param.x2bound;
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
     
%%
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

%% Convex Optimization - compute ROA
cvx_begin sdp
    cvx_solver mosek
    
    % Variables
    variable Q1(nG,nG) symmetric;
    variable Q2(nphi,nphi) diagonal; 
    variable L1(nu, nG); % L1 = \tilde{Nux}*Q1;
    variable L2(nu, nphi); % L2 = \tilde{Nuw}*Q2;
    variable L3(nphi, nG); % L3 = \tilde{Nvx}*Q1;
    variable L4(nphi, nphi); % L4 = \tilde{Nvw}*Q2;
    L = [L1, L2;...
         L3, L4];
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
    rho = param.rho;
    eta_ROA = param.eta_ROA;
    if param.iter == 0
        Yk = ones(nu+nphi, nG+nphi);
    else
        Yk = param.Yk;
    end
    % enforce {x: x'Px<=1} \subset {x: |x1| <= x1bound}
     [1,0]*Q1*[1,0]' <= x1bound^2;
    % enforce {x: x'Px<=1} \subset {x: |x2| <= x2bound}
     [0,1]*Q1*[0,1]' <= x2bound^2;
    % objective function
    obj1 = -eta_ROA*log_det(Q1);
    obj2 = rho/2*pow_pos(norm(fN*Q - L,'fro'), 2);
    obj3 = trace(Yk'*(fN*Q-L));
    obj = obj1 + obj2 + obj3;
%     obj = -eta_ROA*log_det(Q1) + rho/2*pow_pos(norm(fN*Q - L,'fro'), 2) + trace(Yk'*(fN*Q-L));
    minimize(obj)
cvx_end
% save computed decision variables
dec_var.Q1 = Q1;
dec_var.Q2 = full(Q2);
dec_var.L1 = L1;
dec_var.L2 = L2;
dec_var.L3 = L3;
dec_var.L4 = L4;
dec_var.Yk = Yk + rho*(fN*Q - L);
dec_var.W1 = W1;
dec_var.W2 = W2;
dec_var.W3 = W3;
dec_var.obj = [obj1, obj2, obj3];
dec_var.AG = AG;
dec_var.BG = BG;
save([param.path '/' 'sdpvar.mat'], 'dec_var')
end