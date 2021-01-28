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

nG = size(AG, 1);
nu = size(BG, 2);

% Linear discrete-time prediction model
model=LTISystem('A', AG, 'B', BG);

% Input constraints
model.u.min = -2; model.u.max = 2;

% State constraints
model.x.min = [-2.5; -6];
model.x.max = [2.5; 6];

% constraint sets represented as polyhedra
X = Polyhedron('lb',model.x.min,'ub',model.x.max);
U = Polyhedron('lb',model.u.min,'ub',model.u.max);

% Penalties in the cost function
Q = eye(nG);
R = 10;
model.x.penalty = QuadFunction(Q);
model.u.penalty = QuadFunction(R);

% Maximal Invariant Set Computation
[Pinf,Kinf,L] = idare(AG,BG,Q,R);% closed loop system
Acl=AG-BG*Kinf;
S=X.intersect(Polyhedron('H',[-U.H(:,1:nu)*Kinf U.H(:,nu+1)]));
Oinf=max_pos_inv(Acl,S);

model.x.with('terminalSet');
model.x.terminalSet = Oinf;
model.x.with('terminalPenalty');
model.x.terminalPenalty = QuadFunction(Pinf);

% Online MPC object
online_ctrl = MPCController( model, 6 );

% % generate binary search tree
% tree = BinTreePolyUnion(explicit_ctrl.optimizer);
%
% % export control law to C
% tree.toC('primal')
%
% % export control law to C
% explicit_ctrl.optmizer.toC('primal')

% Compute explicit solution
explicit_ctrl = online_ctrl.toExplicit();


% Plot control law (primal solution)
% explicit_ctrl.optimizer.fplot('primal')
if true
    hold on
    num_pt = 100;
    xU = model.x.max;
    xL = model.x.min;
    x1 = linspace(xL(1), xU(1),num_pt);
    x2 = linspace(xL(2),xU(2),num_pt);
    ug = zeros(num_pt,num_pt);
    data = [];
    for i=1:num_pt
        for j = 1:num_pt
            ug(i, j) = online_ctrl.evaluate([x1(i); x2(j)]);
            if ~isnan(ug(i, j))
                data = [data; x1(i), x2(j), ug(i, j)];
            end
        end
    end
    mesh(x1,x2,ug')
    
    % save the state/control data pairs
    writematrix(data,'exp_data2.csv')
end

%% simulate
if false
    Nsim = 300;
    xU = model.x.max;
    xL = model.x.min;
    x01 = linspace(xL(1), xU(1),10);
    x02 = linspace(xL(2),xU(2),10);
    for i = 1:10
        i
        for j = 1:10
            xsim = zeros(nG,Nsim);
            usim = [];
            x0 = [x01(i); x02(j)];
            xsim(:,1) = x0;
            for k =2:Nsim
                ui = online_ctrl.evaluate(xsim(:,k-1));
                if isnan(ui)
                    break
                end
                usim = [usim; ui];
                xsim(:,k) = AG*xsim(:,k-1) + BG*ui;
            end
            if norm(xsim(:,end),2)<0.01 && k == Nsim
                plot(x01(i), x02(j),'o')
                hold on
                plot(xsim(1,:),xsim(2,:))
                hold on
            end
        end
    end
end