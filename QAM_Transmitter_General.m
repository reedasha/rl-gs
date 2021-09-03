function [SignalOut, varargout] = QAM_Transmitter_General(P)
% Ideal Multichannel QAM Transmitter.
% RRC, Low Pass filtering included. Decorrelation between channels
% implemented. Phase noise the same on all channels.
%
% [SignalOut, varargout] = QAM_Transmitter(P)
%
% Inputs:
% P.Fb                      - Channel symbol rate [Hz]
% P.Fs                      - Sample rate [Hz]
% P.Fc                      - Center frequency [Hz]
% P.ModFormat               - Modulation format (needs to be square)
% P.PatternLength           - PRBS length
% P.TransmittedIndices      - Indices of Tx symbols
% P.Channels                - Number of channels (default 1)
%
% Returns:
% SignalOut                 - Output signal structure
% P                         - Parameter structure

%% Get order of constellation, M
str = P.ModFormat;
if strcmp(str(end-2:end),'QAM')
    P.M = str2double(str(1:end-3));
else
    switch str
        case 'QPSK'
            P.M = 4;
        otherwise
            error([str ' not a recognised modulation format'])
    end
end


for Channel = 1:P.Channels
    %% BEGIN CHANGING HERE (to add 8QAM and 32QAM)
    %% Create QAM Signal in Np polarisations
    %% Integer-to-symbol mapping
%     hQAMMod = comm.RectangularQAMModulator('ModulationOrder',P.M,'NormalizationMethod','Average power','SymbolMapping',P.SymbolMapping);
%     
%     if strcmp(P.SymbolMapping,'Custom')
%         if P.verbose == 2
%             disp('Using a custom mapping scheme'); 
%         end
%         hQAMMod.CustomSymbolMapping = P.CustomSymbolMapping;
%     end
    
    %% Output symbols
%     ModulatedSignal.Et(1,:) =   step(hQAMMod, P.TransmittedIndices(2*Channel-1,:).');
%     ModulatedSignal.Et(2,:) =   step(hQAMMod, P.TransmittedIndices(2*Channel,:).');

%     ModulatedSignal.Et(1,:) = qammod(P.TransmittedIndices(2*Channel-1,:).',P.M,P.CustomSymbolMapping,'PlotConstellation',false,'UnitAveragePower',true);
%     ModulatedSignal.Et(2,:) = qammod(P.TransmittedIndices(2*Channel,:).',P.M,P.CustomSymbolMapping,'PlotConstellation',false,'UnitAveragePower',true);
   
    ModulatedSignal.Et(1,:) = genqammod(P.TransmittedIndices(2*Channel-1,:).',P.LabelledConstellation);
    ModulatedSignal.Et(2,:) = genqammod(P.TransmittedIndices(2*Channel,:).',P.LabelledConstellation);
    
       
    P.TransmittedSymbols(2*Channel-1:2*Channel,:) = ModulatedSignal.Et;
    %scatterplot(P.TransmittedSymbols(2*Channel-1,:))
    
    %% END
    
    %% Add modulation parameters to signal struct
    if isfield(P,'Fb')
        ModulatedSignal.Fb = P.Fb;
        ModulatedSignal.Fs = P.Fb;
        ModulatedSignal.Fc = P.Fc;
    end
    
    %
    temp = ModulatedSignal;
    
    %% Upsample and Root Raised Cosine Filtering (RRC)
    P.Ns = ceil(P.Fs/temp.Fb);
    if (isfield(P, 'RRCFilter') && (P.RRCFilter == 1))
        UpSampled = ZeroPad(temp, P);
        MFilter = RRCFilter(UpSampled, P);
        temp = MFilter;
    end

    %% Add channel to SignalOut
    SignalOut = temp;
    SignalOut.Fchan = temp.Fc;
    
end
SignalOut.Fc = P.Fc;
varargout{1} = P;