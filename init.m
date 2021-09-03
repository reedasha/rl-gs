function [points, MI] = getConstellation(M, OSNR)  
    % convert to matlab double
    M = double(M);
    OSNR = double(OSNR);
    
    init = get_MQAM_constellation(M);
    
    %% Normalize Constellation
    P.Constellation = init;
    P.Constellation=P.Constellation-(1/M*sum(P.Constellation));
    P.Constellation=P.Constellation./sqrt(1/M*sum(abs(P.Constellation).^2));

    % Load all other parameters
    P = QBBParameters(M,P);

    [MI] = QAMBlackBox_Single(P,OSNR);
    %% convert complex double to double
    init = P.Constellation;
    
    points = zeros(length(init), 2);
    for i=1:length(init)
        points(i,:) = [real(init(i)),imag(init(i))];
    end
end