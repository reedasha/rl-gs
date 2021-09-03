clc
clear
close all

M = 64;
OSNR = 7;

% Define constellation before all other parameters (IMPORTANT FOR LABELING)
P.Constellation = get_MQAM_constellation(M); % Power normalized constellation

% Normalize Constellation
P.Constellation=P.Constellation-(1/M*sum(P.Constellation));
P.Constellation=P.Constellation./sqrt(1/M*sum(abs(P.Constellation).^2));

% Load all other parameters
P = QBBParameters(M,P);

% Load optimization parameters
% dx = 0.05; 0.25;
% dy = 0.05; 0.25;
% 
% alpha = 0.1; 0.01; 1e-2;

% Save constellation evolution at each iteration step
C = P.Constellation;
% C0 = [];

[GMI0,SNR0,BER0,MI0] = QAMBlackBox_Single(P,OSNR);

%% Initialize parameters
tic
t = 0;
temp = 0.0006;
alpha = 0.9995;
n_iterations = int32(1000);
step_size = 0.1;

temps(1) = temp;
%% convert complex double to double
points = zeros(length(P.Constellation), 2);
for i=1:length(P.Constellation)
    points(i,:) = [real(P.Constellation(i)),imag(P.Constellation(i))];
end

curr = points;
best = curr;
best_mi = MI0;
curr_mi = best_mi;

while 1==1 && t<n_iterations
    t = t+1;
    
    compl_curr = zeros(length(curr), 1);
    for i=1:length(curr)
        compl_curr(i) = complex(curr(i,1), curr(i,2));
    end
    
    C0(:,t) = compl_curr;
    
    % Pick a random point
    randPoint = randi(M);
    
    candidate = curr;
    candidate(randPoint, 1) = curr(randPoint,1) + randn(1)*step_size;
    candidate(randPoint, 2) = curr(randPoint,2) + randn(1)*step_size;
   
    compl = zeros(length(candidate), 1);
    for i=1:length(candidate)
        compl(i) = complex(candidate(i,1), candidate(i,2));
    end
  
    P.Constellation = compl;
    
    % Normalize Constellation
    P.Constellation=P.Constellation-(1/M*sum(P.Constellation));
    P.Constellation=P.Constellation./sqrt(1/M*sum(abs(P.Constellation).^2));
    %figure(10), hold all, plot(P.Constellation,'.')
    
    P = QBBParameters(M,P);
    [GMI,SNR,BER,MI] = QAMBlackBox_Single(P,OSNR);
   
    J0(t) = MI; 
     
    if MI > best_mi
        best_mi = MI;
        best = P.Constellation;
    end
    
    diff = MI - curr_mi;
    
    % multiplicative cooling schedule
    temps(t+1) = temps(t)*alpha;
    
    metropolis(t) = exp(-abs(diff) / temps(t+1));
 
    if diff > 0 || rand <= metropolis(t)
        curr_mi = MI;
        curr = candidate;
    end
    
    figure(7), plot(temps,'ro-','Linewidth',2),xlabel('Iteration'),ylabel('Temperatures')
    figure(6), plot(metropolis,'ro-','Linewidth',2),xlabel('Iteration'),ylabel('Metropolis')
    
    figure(10), hold all, grid on, plot(SNR,MI,'o-','Linewidth',2)
    plot(5:15,log2(1+10.^((5:15)./10)),'k-')
    xlabel('SNR [dB]')
    ylabel('MI [bit/symb.]')
    set(gca,'FontSize',10)
    
    figure(3), clf, grid on, hold all, plot(C0(:,1),'o'), plot(C0(:,end),'x'),legend('Before','After');
    
    figure(2), plot(J0,'ro-','Linewidth',2),xlabel('Iteration'),ylabel('Cost function')
end
disp(best_mi);
figure(3), clf, grid on, hold all, plot(C0(:,1),'o'), plot(best,'x'),legend('Before','After');
toc

