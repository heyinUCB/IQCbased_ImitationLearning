clear
close all
folder = 'data/NN_policy_08-12-2020_16-22-41/';
iter = 19;
NNcolor = 'm';
ROAcolor = 'm';
result_analysis

folder = 'data/NN_policy_04-12-2020_10-48-52/';
NNcolor = mycolor('coolblue');
ROAcolor = mycolor('coolblue');
result_analysis

figure(2)
p = plot3(expert_data(:,1),expert_data(:,2),expert_data(:,3),'r+','MarkerSize',5);
p.Color = mycolor('orange');
legend('NN with $\eta_2 = 5$', 'NN with $\eta_2 = 20$', 'expert data','interpreter','latex')

figure(3)
ROA_LQR(AG,BG,K,x1bound,x2bound)
% Hyperplanes 
X = Polyhedron('lb',[-x1bound; -x2bound],'ub',[x1bound; x2bound]);
X.plot('alpha',0.2,'color',mycolor('orange'),'linewidth',5,'edgecolor',mycolor('maroon'))
hold on
grid on;
axis([-x1bound-0.5, x1bound+0.5, -x2bound-0.5, x2bound+0.5]);
legend('ROA with $\eta_2 = 5$', 'ROA with $\eta_2 = 20$', 'ROA of LQR','$X$','interpreter','latex')
garyfyFigure