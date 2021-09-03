function [GMI, SNR, BER, MI] = QAMBlackBox_Single(P,OSNR)
%% Translate Optical to SI units BEFORE Simulation
P.Fibre = SItoOpt(-1,P.Fibre);
%% Outer Loop indices if required (e.g. for parameter sweeps)

BER = zeros(size(OSNR));
SNR = zeros(size(OSNR));
GMI = zeros(size(OSNR));
MI = zeros(size(OSNR));
% tic

for powerindex = 1:length(OSNR)
    %P.Fibre.PdBm = PdBmvec(powerindex);
    P.Fibre.OSNR = OSNR(powerindex);
    %% Generate QAM
    [TxSignal, P] = QAM_Transmitter_General(P);
    temp = TxSignal;
    % plot_constellations(temp)
    
    %% Channel Model
%     dT = 1/P.Fs;
%     Nt = P.Ns*length(P.TransmittedSymbols);
%     Linewidth = 1e3; % [Hz] 1e3
% 
%     phase_variance=2*pi*dT*Linewidth;
%     phi = sqrt(phase_variance)*cumsum(randn(Nt,1));
% 
%     tempPh(1,:) = reshape(temp.Et(1,:), 524288, 1).*exp(1i*phi);
%     tempPh(2,:) = reshape(temp.Et(2,:), 524288, 1).*exp(1i*phi);
%     temp.Et = tempPh;
    NoiseLoaded = AddNoise(temp, P.Fibre);
    temp = NoiseLoaded;
    for channel_index = 1:P.Channels% Iterates over WDM channels
        if channel_index == ceil(numel(P.Fchan)/2)
            %% Receiver DSP
            P.DSP.FreqOffset        = (P.Fc-P.Fchan(channel_index));
            P.Fibre.spanindex       = 1; %spanindex;
            
            %% Baseband matched filtering
            RRCFiltered = RRCFilter(temp,P.DSP);
            temp = RRCFiltered;

            if ~P.DSP.FullDSP
                temp.Et = temp.Et(:,1:temp.Fs/temp.Fb:end); % Get ideal sampling instant
            end
            
            CarrierRecovered = temp;
            
            %% Set Tx data by channel
            P.BER.TransmittedSymbols    = P.TransmittedSymbols((2*channel_index-1):(2*channel_index),:);
            P.BER.TransmittedIndices    = P.TransmittedIndices((2*channel_index-1):(2*channel_index),:);
            P.BER.CodeIntBits           = P.CodeIntBits((2*channel_index-1):(2*channel_index),:);
            
            if ~P.DSP.FullDSP 
                %% Ideal phase compensation (genie approach)
                
                for index = 1:size(CarrierRecovered.Et,1)
                    %                         figure(channel_index);
                    x=P.BER.TransmittedSymbols(index,:);
                    y=CarrierRecovered.Et(index,:);
                    %X=get_MQAM_constellation(P.M);
                    X=P.LabelledConstellation;
                    %                         subplot(2,1,index);hold on;grid on;axis square;
                    %                         plot(real(X),imag(X),'kx','Linewidth',2,'MarkerSize',10,'MarkerFaceColor','k');
                    %                         plot(real(y),imag(y),'ro','Linewidth',1,'MarkerSize',4,'MarkerFaceColor','r');
                    
                    for ii=1:P.M
                        pnt=find(abs(x-X(ii))<1e-10);   % Find pointers to i-th Tx symbols
                        meany = mean(y(pnt));           % Compute the mean of the Rx symbols
                        phi=angle(meany*X(ii)');        % Find angle
                        y(pnt)=y(pnt)*exp(-1i*phi);     % Rotate the received samples
                        meany = mean(y(pnt));           % Compute mean again
                        hChannel=abs(X(ii))/abs(meany); % Compute gain
                        y(pnt)=y(pnt)*hChannel;         % Scale the received samples
                        %                             plot(real(y(pnt)),imag(y(pnt)),'o','Color',rand(3,1),'Linewidth',1,'MarkerSize',2,'MarkerFaceColor','w');
                        %                             plot(real(X),imag(X),'kx','Linewidth',2,'MarkerSize',10,'MarkerFaceColor','k');
                        N0=var(x(pnt)-y(pnt));
                        SNRdB{channel_index}(index,ii) =10*log10(1/N0);
                        
                    end
                    CarrierRecovered.Et(index,:)=y;
                    
                    %                     figure(100);grid on;hold on;
                    %                     plot([1:P.M],SNRdB{channel_index}(index,:),'o-','Color',rand(3,1));
                    %                     xlabel('Symbol')
                    %                     ylabel('SNR')
                end
            end
            temp=CarrierRecovered;
           
            %% Symbol Demapping and SER/BER Estimation
%             disp(['****************']);
%             disp(['Calculating BER and Q for central channel'])
            [ChannelInfo{channel_index}, P] = QBBMIGMIBER(temp,P); % N.B. Analytical function only

            %%
%             disp(['WDM Channel ' num2str(channel_index)])
%             disp(['****************']);
            %fprintf('OSNR             : %f \n', OSNRvec(powerindex)-10*log10(numel(P.Fchan)))
%             fprintf('BER              : %f \n', mean(ChannelInfo{channel_index}.BER));
%             fprintf('Q                : %f \n', ChannelInfo{channel_index}.Q);
            fprintf('MI               : %f \n', mean(ChannelInfo{channel_index}.MI_AWGN));
            fprintf('GMI              : %1.2f \n', mean(ChannelInfo{channel_index}.GMI_AWGN));
            %fprintf('GMI (Normalized) : %1.2f \n', mean(ChannelInfo{channel_index}.GMI_AWGN)/P.m);
            
            fprintf('mean SNR         : %1.2f \n', mean(ChannelInfo{channel_index}.EsN0dB));
%             fprintf('mean BER         : %1.2f \n', mean(ChannelInfo{channel_index}.BER));
%             fprintf('mean BERLLR      : %1.2f \n', mean(ChannelInfo{channel_index}.BERLLR));
            %disp(['Distance           :  ' num2str(spanindex*totD/Steps) ' km'])
            
            
        end
    end
    
    BER(powerindex) = mean(ChannelInfo{ceil(numel(P.Fchan)/2)}.BER);    
    SNR(powerindex) = mean(ChannelInfo{ceil(numel(P.Fchan)/2)}.EsN0dB);    
    GMI(powerindex) = mean(ChannelInfo{ceil(numel(P.Fchan)/2)}.GMI_AWGN); 
    MI(powerindex)  = mean(ChannelInfo{ceil(numel(P.Fchan)/2)}.MI_AWGN);
end
% toc
%% plot
% figure(1), hold all, plot(SNR,GMI,'o-','Linewidth',2)
% plot(5:15,log2(1+10.^((5:15)./10)),'k-')
% grid on
% xlabel('SNR [dB]')
% ylabel('GMI [bit/symb.]')
% set(gca,'FontSize',10)

