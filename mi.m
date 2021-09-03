function [new_state, reward, MI, done] = getMI(action, const, OSNR, prev_mi)  
    % Convert to matlab   
    const = double(const);
    OSNR = double(OSNR);
    action = double(action)+1;
    prev_mi = double(prev_mi);
    %% Move a point according to an action
    M = length(const);
    step = 0.065;
    
    dir = rem(action, 4);
    
    if dir == 0
        point = fix(action/4);
    else
       point = fix(action/4)+1; 
    end
    
    switch(dir)
        case 1
            const(point,:) = [const(point,1), const(point,2)+step];
        case 2
            const(point,:) = [const(point,1)+step, const(point,2)];
        case 3
            const(point,:) = [const(point,1), const(point,2)-step];
        case 0
            const(point,:) = [const(point,1)-step, const(point,2)];
    end

%     switch(dir)
%         case 1
%             const(point,:) = [const(point,1), const(point,2)+step];
%         case 2
%             const(point,:) = [const(point,1)+step, const(point,2)+step];
%         case 3
%             const(point,:) = [const(point,1)+step, const(point,2)];
%         case 4
%             const(point,:) = [const(point,1)+step, const(point,2)-step];
%         case 5
%             const(point,:) = [const(point,1), const(point,2)-step];
%         case 6
%             const(point,:) = [const(point,1)-step, const(point,2)-step];
%         case 7
%             const(point,:) = [const(point,1)-step, const(point,2)];
%         case 0
%             const(point,:) = [const(point,1)-step, const(point,2)+step];
%     end
    %% 
    c = zeros(length(const), 1);
    for i=1:length(const)
        c(i) = complex(const(i,1), const(i,2));
    end
    
    P.Constellation = c;
    
    % Normalize Constellation
    P.Constellation=P.Constellation-(1/M*sum(P.Constellation));
    P.Constellation=P.Constellation./sqrt(1/M*sum(abs(P.Constellation).^2));

    figure(1),  plot(P.Constellation,'ro', 'LineWidth', 2), grid on
    
    % Load all other parameters
    P = QBBParameters(M,P);
    
    [GMI,SNR,BER,MI] = QAMBlackBox_Single(P,OSNR);
    
    %% Return reward, give the difference reward
    
    reward = (MI - prev_mi);
    
    if reward > 0
        reward = reward*10;
    end

%     if MI >= prev_mi
%         reward = 1;
%     else
%         reward = 0;
%     end
    %% Check if MI is lower than the threshold
    thresh = 2;
    
    if MI <= thresh
        done = true;
    else
        done = false;
    end
    %% convert complex double to double
    new_state = zeros(length(P.Constellation), 2);
    for i=1:length(P.Constellation)
        new_state(i,:) = [real(P.Constellation(i)),imag(P.Constellation(i))];
    end
end