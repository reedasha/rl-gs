function [ ChannelInfo, P ] = QBBMIGMIBER( SignalIn , P )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ReceivedSymbols]   = SymbolAlignment(SignalIn.Et,P.BER);
% hQAMDemod           = comm.RectangularQAMDemodulator('ModulationOrder',P.M,'NormalizationMethod','Average power','SymbolMapping',P.SymbolMapping);
% if strcmp(P.SymbolMapping,'Custom')
%     hQAMDemod.CustomSymbolMapping = P.CustomSymbolMapping;
% end
% keyboard
ReceivedIndices   = zeros(size(SignalIn.Et));
SymbolErrors      = zeros(size(SignalIn.Et));
SER               = zeros(size(SignalIn.Et));
Receivedbits      = zeros(size(P.CodeIntBits));
BitErrors         = zeros(size(P.CodeIntBits)) ;
BER               = zeros(size(SignalIn.Et,1),1);
BERPerPosition    = zeros(size(SignalIn.Et,1),log2(P.M));

EsN0dB   = zeros(size(SignalIn.Et,1),1);
MI_AWGN  = zeros(size(SignalIn.Et,1),1);
GMI_AWGN = zeros(size(SignalIn.Et,1),1);
GMI      = zeros(size(SignalIn.Et,1),1);
BERLLR   = zeros(size(SignalIn.Et,1),1);

disp('num of polarizations')
disp(size(SignalIn.Et,1))
% Loop over polarizations
for index = 1:size(SignalIn.Et,1)
    % SER Estimation
%     ReceivedIndices(index,:)= step(hQAMDemod, ReceivedSymbols(index,:).')';
%     ReceivedIndices(index,:)= qamdemod(ReceivedSymbols(index,:).',P.M,P.CustomSymbolMapping)';
    ReceivedIndices(index,:)= genqamdemod(ReceivedSymbols(index,:).',P.LabelledConstellation)';
    %keyboard
    SymbolErrors(index,:) 	= ReceivedIndices(index,:)~=P.BER.TransmittedIndices(index,:);
    SER(index,:)            = mean(SymbolErrors(index,P.BER.SDiscard:(end-P.BER.EDiscard)));
    % Raw BER Estimation (average)
    hIntToBit               = comm.IntegerToBit(log2(P.M));
    Transmittedbits         = P.BER.CodeIntBits(index,:)';
    Receivedbits(index,:)   = step(hIntToBit,ReceivedIndices(index,:)');
    BitErrors(index,:)      = xor(Transmittedbits,Receivedbits(index,:).');
    BER(index,:)            = mean(BitErrors(index,P.BER.SDiscard:(end-P.BER.EDiscard)));
    % Raw BER Estimation (per bit position)
    BitErrorsPerPosition    = reshape(BitErrors(index,:),log2(P.M),length(BitErrors)/log2(P.M));
    BERPerPosition(index,:) = mean(BitErrorsPerPosition(:,P.BER.SDiscard:(end-P.BER.EDiscard)),2).';
    x=P.BER.TransmittedSymbols(index,P.BER.SDiscard+1:end-P.BER.EDiscard).';
    y=ReceivedSymbols(index,P.BER.SDiscard+1:end-P.BER.EDiscard).';
    
    N0=var(x-y);
    EsN0dB(index,:) =10*log10(1/N0);
    num_of_syms=size(x,1);

    x=[real(x),imag(x)];
    y=[real(y),imag(y)];
    % LLR calculation (assumes an AWGN channel)
    IntLLR             	= demapper_mex(P.X,P.Lbin,EsN0dB(index,:),y,P.Ik1,P.Ik0,1);
    IntCodeBits         = P.BER.CodeIntBits(index,P.BER.SDiscard*log2(P.M)+1:end-P.BER.EDiscard*log2(P.M));    
    %IntCodeBits         = P.BER.CodeIntBits(index,P.BER.SDiscard*log2(P.M)+1:end-P.BER.EDiscard*log2(P.M));    
    hat_IntCodeBits     = 1*(IntLLR(:)>0);                                          % Hard-decision based on L-values
    BERLLR(index,:)     = sum(IntCodeBits~=hat_IntCodeBits.')/length(IntCodeBits);  % BER based on LLRs
    
    %% MI Calculation based on Monte Carlo integration, i.e., it assumes AWGN channel
    MI_AWGN(index,:)            = MI_uniform_MonteCarlo_mex(x,y,N0);
    %MI(index,:)=0;              % No channel model assumption, Tobias Fehenberger's approach, only for 1D
	%% GMI Calculation based on (4.86) and (4.90) of our book (Monte Carlo using LLRs)
    GMI_AWGN(index,:)=0;
    for kk=1:P.m
       GMI_AWGN(index,:)= (1-1/(num_of_syms)*sum(log2(1+exp((-1).^(IntCodeBits(kk:P.m:end).').*IntLLR(kk,:).')))) + GMI_AWGN(index,:);
    end
    
    GMI(index,:)=0; 

end

% Add BER, SERs, MIs, GMIs, etc.
ChannelInfo.BERPerPosition  = BERPerPosition;
ChannelInfo.BER             = BER;
ChannelInfo.Q               = 20*log10(sqrt(2)*erfcinv(2*mean(BER)));
ChannelInfo.SER             = SER;
ChannelInfo.MI_AWGN         = MI_AWGN;
%ChannelInfo.MI              = MI;
ChannelInfo.GMI_AWGN        = GMI_AWGN;
ChannelInfo.GMI             = GMI;
ChannelInfo.BERLLR          = BERLLR;
ChannelInfo.EsN0dB          = EsN0dB;

% Add Rx indices and symbols to P.BER
P.BER.ReceivedIndices=ReceivedIndices;
P.BER.ReceivedSymbols=ReceivedSymbols;
P.BER.Receivedbits=Receivedbits;

% if P.FEC.CoDec>0 % If FEC implemented
%     ChannelInfo.PosFECBER  	= PosFECBER;
% end % EOF

