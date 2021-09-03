clc
clear
clear classes
close all
% terminate(pyenv)
% pyenv("Version","/home/reedvl/anaconda3/bin/python", "ExecutionMode", "OutOfProcess")

M = int32(16);
OSNR = 6;

% Define constellation before all other parameters (IMPORTANT FOR LABELING)
% P.Constellation = get_MQAM_constellation(M); % Power normalized constellation

% Normalize Constellation
% P.Constellation=P.Constellation-(1/M*sum(P.Constellation));
% P.Constellation=P.Constellation./sqrt(1/M*sum(abs(P.Constellation).^2));

% Load all other parameters
% P = QBBParameters(M,P);

% Load optimization parameters
% dx = 0.05; 0.25;
% dy = 0.05; 0.25;
% 
% alpha = 0.1; 0.01; 1e-2;

% Save constellation evolution at each iteration step
% C = P.Constellation;
% C0 = [];

% [GMI0,SNR0,BER0,MI0] = QAMBlackBox_Single(P,OSNR);

%% Run gradient descent optimizer
tic
n_episodes=int32(70);
max_t=int32(100);
gamma=1.0;
print_every=int32(1);

 %% set up python path
% terminate(pyenv)
pyversion
mod = py.importlib.import_module('RL');
py.importlib.reload(mod); %reload modified python module

pathToRL = fileparts(which('RL.py'));
if count(py.sys.path,pathToRL) == 0
   insert(py.sys.path,int32(0),pathToRL);
end
%% Call RL
% terminate(pyenv)
py.RL.reinforce(OSNR, M, n_episodes, max_t, gamma);

% figure(7), plot(temps,'ro-','Linewidth',2),xlabel('Iteration'),ylabel('Temperatures')
% figure(6), plot(metropolis,'ro-','Linewidth',2),xlabel('Iteration'),ylabel('Metropolis')
% 
% figure(10), hold all, grid on, plot(SNR,best_mi,'o-','Linewidth',2)
% plot(5:15,log2(1+10.^((5:15)./10)),'k-')
% xlabel('SNR [dB]')
% ylabel('MI [bit/symb.]')
% set(gca,'FontSize',10)
% 
% figure(3), clf, grid on, hold all, plot(C0(:,1),'o'), plot(C0(:,end),'x'),legend('Before','After');
% figure(2), plot(J0,'ro-','Linewidth',2),xlabel('Iteration'),ylabel('Cost function')
% 
% disp(best_mi);
% figure(3), clf, grid on, hold all, plot(C0(:,1),'o'), plot(best,'x'),legend('Before','After');
toc

