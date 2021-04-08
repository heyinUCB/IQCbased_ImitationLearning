clear
close all
% initial condition for simulation
% [x0in, x0on]=psample(Vval-1,x,zeros(nG,1),1);
x0in = [0.296; -0.1464;0.2066;0.0114];
num_sim = 500;

% load the ROA for MPC
% eta_ROA = 100
folder = 'data/NN_policy_22-03-2021_21-47-43/';
iter = 19;
ROAcolor = 'c';
simcolor = 'c';
result_analysis

% eta_ROA = 500
folder = 'data/NN_policy_23-03-2021_12-25-03/';
ROAcolor = 'b';
simcolor = 'b';
result_analysis

% load the ROA for the MPC
load ROA_MPC.mat
figure(1)
subplot(1,2,1)
for i = 1:size(ROAMPC_x1x2,1)
    plot(ROAMPC_x1x2(i,1),ROAMPC_x1x2(i,2),'rx')
    hold on
end
subplot(1,2,2)
for i = 1:size(ROAMPC_x3x4,1)
    plot(ROAMPC_x3x4(i,1),ROAMPC_x3x4(i,2),'rx')
    hold on
end
legend('$\eta_2=100$','$\eta_2=500$','MPC','interpreter','latex')
garyfyFigure

% simulation
xMPC = [x0in];
xMPCcurr = x0in;
uMPC = [];
load MPC_control.mat
timer = tic;
for i = 1:num_sim
    uMPCcurr = explicit_ctrl.evaluate(xMPCcurr);
    xMPCnext = AG*xMPCcurr + BG*uMPCcurr;
    xMPC = [xMPC, xMPCnext];
    uMPC = [uMPC, uMPCcurr];
    xMPCcurr = xMPCnext;
end
toc(timer)
dt = 0.02;
figure(2)
subplot(5,1,1)
plot(dt*(0:num_sim), xMPC(1,:),'r')
set(gca,'xticklabel',[])

subplot(5,1,2)
plot(dt*(0:num_sim), xMPC(2,:),'r')
set(gca,'xticklabel',[])

subplot(5,1,3)
plot(dt*(0:num_sim), xMPC(3,:),'r')
set(gca,'xticklabel',[])

subplot(5,1,4)
plot(dt*(0:num_sim), xMPC(4,:),'r')
set(gca,'xticklabel',[])

subplot(5,1,5)
plot(dt*(0:num_sim-1), uMPC,'r')
xlabel('$t$','interpreter','latex')
legend('$\eta_2=100$','$\eta_2=500$','MPC','interpreter','latex')
garyfyFigure