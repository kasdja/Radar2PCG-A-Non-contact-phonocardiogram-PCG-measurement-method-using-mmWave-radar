%% ========================================================================
% A Deep Learning-based Non-contact Phonocardiogram Measurement Method
% Using mmWave RadarA Deep Learning-based Non-contact Phonocardiogram 
% Measurement Method Using mmWave Radar 
% 
%
% Signal Processing Method for Cardiac Mechanical Activity Extraction
% ========================================================================
%
% Author:  Haozhe Liu
% Email:   2021117151@stumail.nwu.edu.cn
% Date:    March 2026
%
% Description:
%   This repository contains the MATLAB implementation of the cardiac signal 
%   extraction Method described in our paper:
%
%   Haozhe Liu, et al., "A Deep Learning-based Non-contact Phonocardiogram 
%   Measurement Method Using mmWave RadarA Deep Learning-based Non-contact 
%   Phonocardiogram Measurement Method Using mmWave Radar ," 2026.
%   
%   (Under Review)
% Contents:
%   - This part of the code is used for basic analysis of radar data
%
% Hardware:
%   - TI IWR1843 mmWave Radar + DCA1000EVM
%   - Eko Core 500 Digital Stethoscope (reference device)
%
% License:
%   This code is released under the MIT License.
%   You are free to use, modify, and distribute this code, provided that
%   proper citation is given to the original paper.
%
% Disclaimer:
%   This code is provided for academic and research purposes only. 
%   It is NOT intended for clinical diagnosis or medical decision-making. 
%   The authors assume no responsibility for any consequences arising 
%   from the use of this code.
%
%   This is a partial release accompanying a manuscript currently under 
%   peer review. The full codebase will be made available upon paper acceptance.
%
% ========================================================================
clc;
clear;
close all;
%% Radar parameter settings 
numADCSamples = 256;
numADCBits = 16;
numTX=3;
numRX = 4; 
numLanes = 2; 
isReal = 0; 
chirpLoop = 1; 
Fs=5e6;
c=3*1e8; 
ts=numADCSamples/Fs;
slope=64.985e12;
B_valid =ts*slope;
detaR=c/(2*B_valid);
startFreq = 77.00e9; 
lambda=c/startFreq;
Tr = 60e-6;
Idle_time = 10e-6;
Tc = Tr+Idle_time;
Bandwidth = slope * Tr ;
BandwidthValid = numADCSamples/Fs*slope;
Range_Res=c/(2*BandwidthValid);
%% Data preprocessing
Filename = 'DATA\0\0.bin';
fid = fopen(Filename,'r');
adcDataRow= fread(fid, 'int16');
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    adcDataRow(adcDataRow > l_max) = adcDataRow(adcDataRow > l_max) - 2^numADCBits;
end
fclose(fid);
process_num = 2000;                   
fileSize=process_num*chirpLoop*numADCSamples*numTX*numRX*2;  
PRTnum = fix(fileSize/(numADCSamples*numRX));
fileSize = PRTnum * numADCSamples*numRX;
adcData = adcDataRow(1:fileSize);

if isReal
    numChirps = fileSize/numADCSamples/numRX;
    LVDS = zeros(1, fileSize);
    LVDS = reshape(adcData, numADCSamples*numRX, numChirps);
    LVDS = LVDS.';
else
    numChirps = fileSize/2/numADCSamples/numRX;  
    LVDS = zeros(1, fileSize/2);
    counter = 1;
    for i=1:4:fileSize-1
        LVDS(1,counter) = adcData(i) + sqrt(-1)*adcData(i+2);   
        LVDS(1,counter+1) = adcData(i+1)+sqrt(-1)*adcData(i+3); 
        counter = counter + 2;
    end
end
%% The radar data is divided into 12 time-division multiplexed (TDM) signals: numTX × numRX.
store_base = zeros(numTX*numRX, numADCSamples*numChirps/numTX);
counter = 1;
r = 1;
adcnum = 1;
is_break = 0;
while(r~=fileSize/2)
    for i = 1:numTX
        for j = 1:numRX
            store_base(counter,adcnum:adcnum+numADCSamples-1) = LVDS(1,r:r+numADCSamples-1);
            counter = counter + 1;
            r = r+numADCSamples;
            if(r==fileSize/2+1)
                is_break = 1;
            end
        end
        if(is_break)
            break;
        end
    end
    if(is_break)
        break;
    end
    counter = 1;
    adcnum =adcnum + numADCSamples;
end
%% Reorganize the data into 12 × ADC × number of chirps transmitted per TX antenna.
processed_adc = zeros(12, numADCSamples, numChirps/numTX);
lopp = 1;
for i = 1:12
    for j = 1:numChirps/numTX
        processed_adc(i,1:256,j)=store_base(i,lopp:lopp+255);
        lopp = lopp+256;
    end
    lopp = 1;
end
%% Range-FFT
RangeFFT = numADCSamples;
DopplerFFT = numChirps/numTX;

range_win = hamming(numADCSamples);
range_win = range_win.';

range_profile = zeros(12,numADCSamples,numChirps/numTX);

for k=1:12
   for m=1:numChirps/numTX
      temp=processed_adc(k,:,m);
      temp_fft=fft(temp,RangeFFT);
      range_profile(k,:,m)=temp_fft;
   end
end
%% Dopplor-FFT
dopplor_win = hamming(numChirps/numTX); 
hamming_window_3d = reshape(dopplor_win, [1, 1, numChirps/numTX]);
speed_profile = zeros(12,numADCSamples,numChirps/numTX);
for k=1:12
    for n=1:RangeFFT
      temp=range_profile(k,n,:).*hamming_window_3d;
      temp_fft=fftshift(fft(temp,DopplerFFT));
      speed_profile(k,n,:)=temp_fft;  
    end
end

%% Angle-FFT
angleFFT = 180;
angle_profile = zeros(angleFFT,numADCSamples,numChirps/numTX);
angle_profile_e = zeros(angleFFT,numADCSamples,numChirps/numTX);
for n=1:RangeFFT   %range
    for m=1:DopplerFFT %chirp
      temp=speed_profile(:,n,m);   
      temp=temp([1,2,3,4],:);
      temp_fft=fftshift(fft(temp,angleFFT)); 
      angle_profile(:,n,m)=temp_fft;  
    end
end
for n=1:RangeFFT   %range
    for m=1:DopplerFFT %chirp
      temp=speed_profile(:,n,m);   
      temp=temp([3,5],:);
      temp_fft=fftshift(fft(temp,angleFFT));   
      angle_profile_e(:,n,m)=temp_fft;  
    end
end
%% Paints 2D-FFT
figure(1);
speed_profile_temp = reshape(speed_profile(1,:,:),RangeFFT,DopplerFFT);   
speed_profile_Temp = speed_profile_temp';
[X,Y]=meshgrid((0:RangeFFT-1)*Fs*c/RangeFFT/2/slope,(-DopplerFFT/2:DopplerFFT/2-1)*lambda/Tc/DopplerFFT/2);
speed_profile_Temp_abs = abs(speed_profile_Temp);
mesh(X,Y,(abs(speed_profile_Temp)));
xlabel('Range(m)');ylabel('Speed(m/s)');zlabel('Signal Amplitude');
title('2D FFT Processing of 3D Views');
xlim([0 (RangeFFT-1)*Fs*c/RangeFFT/2/slope]); 
ylim([(-DopplerFFT/2)*lambda/Tc/DopplerFFT/2 (DopplerFFT/2-1)*lambda/Tc/DopplerFFT/2]);
%% Paints RA-heatmap
Range_Index=Range_Res*(1:RangeFFT);
Speed_Res=lambda/(2*DopplerFFT*Tc);
Speed_Index=(-DopplerFFT/2:1:DopplerFFT/2-1)*Speed_Res;
Azimuth_Index=(-angleFFT/2:1:angleFFT/2-1);
angle_profile = permute(angle_profile,[2,3,1]);

w = linspace(-1,1,angleFFT); % angle_grid
agl_grid = asin(w)*180/pi; % [-1,1]->[-pi/2,pi/2]

Nr = size(angle_profile,1);   %%%length of Chirp(num of rangeffts)
Ne = size(angle_profile,3);   %%%number of angleffts
Nd = size(angle_profile,2);   %%%length of chirp loop

Xpow = abs(angle_profile);
Xpow = squeeze(sum(Xpow,2)/size(Xpow,2));

Xsnr = Xpow;
% Xsnr = pow2db(Xpow/noisefloor);
heatmap_data = Xpow/50000;

figure(2);
set(gcf,'Position',[10,10,530,420])
[axh] = surf(agl_grid,Range_Index,heatmap_data);
view(0,90)
xlim([-60 60]);
ylim([0.2,3]);
grid off
shading interp
colorbar
clim([0,15])
title('Range-Azimuth Heatmap', 'FontWeight', 'bold');
xlabel('Azimuth(°)');
ylabel('Range(m)');
%% Paints RE-heatmap
angle_profile_e = permute(angle_profile_e,[2,3,1]);
Nr = size(angle_profile_e,1);   %%%length of Chirp(num of rangeffts)
Ne = size(angle_profile_e,3);   %%%number of angleffts
Nd = size(angle_profile_e,2);   %%%length of chirp loop

Xpow = abs(angle_profile_e);
Xpow = squeeze(sum(Xpow,2)/size(Xpow,2));
Xsnr = Xpow;
% Xsnr = pow2db(Xpow/noisefloor);
heatmap_data_e = Xpow/50000;

figure(3);
set(gcf,'Position',[10,10,530,420])
[axh] = surf(agl_grid,Range_Index,heatmap_data_e);
view(0,90)
xlim([-60 60]);
ylim([0.2,3]);
grid off
shading interp
colorbar
clim([0,15])
title('Range-Elevation Heatmap', 'FontWeight', 'bold');
xlabel('Elevation(°)');
ylabel('Range(m)');