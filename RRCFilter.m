function [SignalOut, varargout] = RRCFilter(SignalIn, P)

% Root Raised Cosine Optical filter.
%
% SignalOut=RRcosineOpticalFilter(SignalIn,P)
%
% Inputs:
% Signal                - Input signal structure
% Signal.Fs             - Sampling rate [Hz]
% Signal.Fb             - Symbol rate [Hz]
% P.RRCOffset           - Frequency offset for the RRC filter [Hz]
% P.RollOff             - Roll off of raised cosine
% P.RRCType             - Implementation type 'Ideal' 'Taps' 'Att'
%
% Optional Inputs:
% P.RRCTaps             - Number of filter for type 'Taps'
% P.RRCAtt              - Stop band attenuation for type 'Taps'
%
% Returns:
% Signal - output signal structure
% P.RRCFilter           - filter transfer function

SignalOut = SignalIn;

% If number of samples per symbol is less than 2 throw and error.
if(SignalIn.Fs/SignalIn.Fb < 2 )
    error('Number of samples per symbol is less than 2!')
end

[Np,Nt] = size(SignalIn.Et);                  % Total number of points
dF = (SignalIn.Fs)/Nt;                        % Spectral resolution [Hz]
FF = MakeTimeFrequencyArray(SignalIn);

if ~isfield(P, 'RollOff'),
    P.RollOff = 0.5;
end

if ~isfield(P, 'RRCOffset'),
    P.RRCOffset = 0;
end

switch P.RRCType
    case 'Ideal'
        Ef = ifft(SignalIn.Et,[],2);
        
        % Centre flat top of frequency responce
        IndTop = abs(FF)<= (1-P.RollOff)*SignalIn.Fb/2;
        
        % Rising and falling edges of filer frequency responce
        IndEdges = find( (abs(FF)>(1-P.RollOff)*SignalIn.Fb/2) & (abs(FF)<=(1+P.RollOff)*SignalIn.Fb/2) );
        
        % Frequecy components beyond excess bandwidth
        % IndZeros = find(abs(FF) > (1+P.RollOff)*Signal.Fb/2);
        
        % Filter frequency response
        hh = zeros(1,Nt);
        hh(IndTop) = 1;
        hh(IndEdges) = 0.5*(1+cos( pi/(P.RollOff*SignalIn.Fb).*( abs(FF(IndEdges))-((1-P.RollOff)*SignalIn.Fb/2))));
        
        % Root Rasied Cosine Response
        hh = sqrt(hh);
        
        ind = round(P.RRCOffset/dF);
        hh = circshift(hh, [1 -ind]);
        
        if (Np == 2)
            hh = [hh; hh];
        end
        
        Gf = hh.*Ef;
        
        SignalOut.Et = fft(Gf,[],2);
    case 'Taps'
        if ~isfield(P, 'RRCTaps'),
            P.RRCTaps = 30;
        end
        
        D = fdesign.pulseshaping(SignalIn.Fs/SignalIn.Fb, 'Square Root Raised Cosine', 'N,Beta', P.RRCTaps, P.RollOff);
        hh = design(D);
        Offset = P.RRCOffset*pi/SignalIn.Fb;
        hh.Numerator = hh.Numerator.*exp(1j*Offset*(1:length(hh.Numerator)));
        
        SignalOut.Et(1,:) = filter(hh, SignalOut.Et(1,:));
        SignalOut.Et(2,:) = filter(hh, SignalOut.Et(2,:));
    case 'Att'
        if ~isfield(P, 'RRCAtt'),
            P.RRCAtt = 30;
        end
        
        D = fdesign.pulseshaping(SignalIn.Fs/SignalIn.Fb, 'Square Root Raised Cosine', 'Ast,Beta', P.RRCAtt, P.RollOff);
        hh = design(D);
        Offset = P.RRCOffset*pi/SignalIn.Fb;
        hh.Numerator = hh.Numerator.*exp(1j*Offset*(1:length(hh.Numerator)));
        
        SignalOut.Et(1,:) = filter(hh, SignalOut.Et(1,:));
        SignalOut.Et(2,:) = filter(hh, SignalOut.Et(2,:));
end 

P.RRCFilter = hh;
varargout{1} = P;

%% Verbose
if(isfield(P, 'verbose') && (P.verbose==1))
    %disp(['RRC Filtering, Type: ' P.RRCType]);
    %disp(['RRC Filter RollOff: ' num2str(P.RollOff)]);
    if strcmp(P.RRCType,'Taps')
        disp(['RRC Filter with ' num2str(P.RRCTaps) ' Taps']);
    elseif strcmp(P.RRCType,'Att')
        disp(['RRC Filter with ' num2str(P.RRCAtt) 'dB stopband attenuation']);
    end
    %disp(['RRC Offset: ' num2str(P.RRCOffset/1e9) 'GHz']);
end
