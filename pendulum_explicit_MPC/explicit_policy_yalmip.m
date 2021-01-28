clear
% parameters
g = 10; % gravitational coefficient
m = 0.15; % mass
l = 0.5; % length
mu = 0.05; % frictional coefficient
dt = 0.02; % sampling period

AG = [1,      dt;...
      g/l*dt, 1-mu/(m*l^2)*dt];
BG = [0; dt/(m*l^2)];

% AG = 1.1*[1, 1; 0, 1];
% BG = [1; 0.5];
nG = size(AG, 1);
nu = size(BG, 2);

% Linear discrete-time prediction model
model=LTISystem('A', AG, 'B', BG);

% Input constraints
model.u.min = -1; 
model.u.max = 1;

% State constraints
model.x.min = [-1; -3];
model.x.max = [1; 3];

% constraint sets represented as polyhedra
X = Polyhedron('lb',model.x.min,'ub',model.x.max);
U = Polyhedron('lb',model.u.min,'ub',model.u.max);

% Penalties in the cost function
Q = eye(nG);
R = 0.1;
N = 6;

% Maximal Invariant Set Computation
[Finf,Pinf]=dlqr(AG,BG,Q,R);
% closed loop system
Acl=AG-BG*Finf;
S=X.intersect(Polyhedron('H',[-U.H(:,1:nu)*Finf U.H(:,nu+1)]));
Oinf=max_pos_inv(Acl,S);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MPC control
options = sdpsettings('solver','quadprog');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup the CFTOC
x = sdpvar(nG,N+1);
u = sdpvar(nu,N);
%set terminal constraint
constr = Oinf.H(:,1:nG)*x(:,N+1)<=Oinf.H(:,nG+1);
%set terminal cost
cost = x(:,N+1)'*Pinf*x(:,N+1);
for k = 1:N
constr = [constr, x(:,k+1) == AG*x(:,k) + BG*u(:,k),...
model.u.min <= u(:,k),u(:,k) <= model.u.max,...
model.x.min <= x(:,k+1),x(:,k+1)<=model.x.max];
cost = cost + x(:,k)'*Q*x(:,k) + u(:,k)'*R*u(:,k);
end
[sol,diagnost,Uz,J,Optimizer] = solvemp(constr,cost,[],x(:,1));
plot(Optimizer)