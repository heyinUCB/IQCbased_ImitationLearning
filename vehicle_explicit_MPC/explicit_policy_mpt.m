clear

%% parameters
% Nominal speed of the vehicle travels at.
U = 28; % m/s
% Model
% Front cornering stiffness for one wheel.
Ca1 = -61595; % unit: Newtons/rad
% Rear cornering stiffness for one wheel.
Ca3 = -52095; % unit: Newtons/rad

% Front cornering stiffness for two wheels.
Caf = Ca1*2; % unit: Newtons/rad
% Rear cornering stiffness for two wheels.
Car = Ca3*2; % unit: Newtons/rad

% Vehicle mass
m = 1670; % kg
% Moment of inertia
Iz = 2100; % kg/m^2

% Distance from vehicle CG to front axle
a = 0.99; % m
% Distance from vehicle CG to rear axle
b = 1.7; % m

g = 9.81;

% sampling period
dt = 0.02;

% Continuous-time state space matrices
% States are lateral displacement(e) and heading angle error(deltaPsi) and 
% their derivatives.
% Inputs are front wheel angle and curvature of the road.
Ac = [0 1 0 0; ...
      0, (Caf+Car)/(m*U), -(Caf+Car)/m, (a*Caf-b*Car)/(m*U); ...
      0 0 0 1; ...
      0, (a*Caf-b*Car)/(Iz*U), -(a*Caf-b*Car)/Iz, (a^2*Caf+b^2*Car)/(Iz*U)];
Bc = [0; 
      -Caf/m;
      0; ...
      -a*Caf/Iz];
AG = Ac*dt + eye(4);
BG = Bc*dt;

nG = size(AG, 1);
nu = size(BG, 2);

%% Linear discrete-time prediction model
model=LTISystem('A', AG, 'B', BG);

% Input constraints
model.u.min = -pi/6; model.u.max = pi/6;

% State constraints
model.x.min = [-2; -5; -1; -5];
model.x.max = [2; 5; 1; 5];

% constraint sets represented as polyhedra
X = Polyhedron('lb',model.x.min,'ub',model.x.max);
U = Polyhedron('lb',model.u.min,'ub',model.u.max);

% Penalties in the cost function
Q = diag([1, 5, 0.1, 0.5]);
R = 0.1;
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
online_ctrl = MPCController( model, 5 );

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
save MPC_control.mat explicit_ctrl
% Plot control law (primal solution)
% explicit_ctrl.optimizer.fplot('primal')

% generate date pairs for training
if false
    xU = model.x.max;
    xL = model.x.min;
    x1 = linspace(xL(1), xU(1),20);
    x2 = linspace(xL(2), xU(2),30);
    x3 = linspace(xL(3), xU(3),20);
    x4 = linspace(xL(4), xU(4),30);
    ug = zeros(20,30,20,30);
    data = [];
    for ii =1:20
        ii
        for jj = 1:30
            for kk = 1:20
                for ll = 1:30
                    ug(ii, jj, kk, ll) = online_ctrl.evaluate([x1(ii); x2(jj); x3(kk); x4(ll)]);
                    if ~isnan(ug(ii, jj, kk, ll))
                        data = [data; x1(ii), x2(jj), x3(kk), x4(ll), ug(ii, jj, kk, ll)];
                    end
                end
            end
        end
    end
    
    % save the state/control data pairs
    writematrix(data,'exp_data.csv')
end

%% simulate to plot ROA
if false
    Nsim = 1000;
    xU = model.x.max;
    xL = model.x.min;
    num_pt = 20;
    x01 = linspace(xL(1), xU(1),num_pt);
    x02 = linspace(xL(2), xU(2),num_pt);
    ROAMPC_x1x2 = [];
    for i = 1:num_pt
        i
        for j = 1:num_pt
            xsim = zeros(nG,Nsim);
            usim = [];
            x0 = [x01(i); x02(j); 0; 0];
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
                ROAMPC_x1x2 = [ROAMPC_x1x2; x01(i), x02(j)];
                subplot(1,2,1)
                plot(x01(i), x02(j),'ro')
                hold on
            end
        end
    end
end

if false
    Nsim = 1000;
    xU = model.x.max;
    xL = model.x.min;
    num_pt = 20;
    x03 = linspace(xL(3), xU(3),num_pt);
    x04 = linspace(xL(4), xU(4),num_pt);
    ROAMPC_x3x4 = [];
    for i = 1:num_pt
        i
        for j = 1:num_pt
            xsim = zeros(nG,Nsim);
            usim = [];
            x0 = [0; 0; x03(i); x04(j)];
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
                ROAMPC_x3x4 = [ROAMPC_x3x4; x03(i), x04(j)];
                subplot(1,2,2)
                plot(x03(i), x04(j),'ro')
                hold on
            end
        end
    end
    save ROA_MPC ROAMPC_x1x2 ROAMPC_x3x4
end