function plotmarkers(Data,Speed)

%% PLOTMARKERS
%
% plotmarkers plots an animation of a 2D markers
%
% SYNTAX:
%
%      plotmarkers(DataM,Speed)
%
%
% Inputs: DATA, Speed
%
% DataM: structure generated from 'csv2strc'.
% Speed:   playback speed as Frames Per Iteration. Most of the times, for a
%        "Real time" playback speed, this parameter should be set on 8.
%        This number could be greater or lower according to the computer's 
%        graphic performance.
%
%
% =========================================================================
%
% This software is part of the public domain, and may therefore be
% distributed freely for any non-commercial use. Contributions,
% improvements, and especially bug fixes of the code are more than welcome.
%
% [V0.1], Juan Pablo Angel Lopez (jangel@autonoma.edu.co), February 2020.
% For more information, see the <a href="matlab: web('https://www.mathworks
% .com/matlabcentral/fileexchange/63721-csv-files-processing-for-mocap-with
% -optitrack')">MathWorks File Exchange Web site</a>.

NSets = numel(fieldnames(Data.Sets));

NMarkersT = 0;
MSets = fieldnames(Data.Sets);

for s=1:NSets
    SetName = MSets{s};
    NMarkersT = NMarkersT + Data.Sets.(sprintf('%s',char(SetName))).NMarkers;
end

Markers1 = zeros(Data.Frames,NMarkersT,3);
counter = 0;

for s=1:NSets
    SetName = MSets{s};
    SetNMarkers = Data.Sets.(sprintf('%s',char(SetName))).NMarkers;
    Labels = fieldnames(Data.Sets.(sprintf('%s',char(SetName))).Raw);
    label = 1;
    for M=counter+1:counter+SetNMarkers
        Markers1(:,M,1) = Data.Sets.(sprintf('%s',char(SetName))).Raw.(sprintf('%s',char(Labels(label))))(:,1);
        Markers1(:,M,2) = Data.Sets.(sprintf('%s',char(SetName))).Raw.(sprintf('%s',char(Labels(label))))(:,2);
        label = label+1;
    end
    counter = M;
end



ejes = [0 500 0 500];


for f=1:Speed:Data.Frames
    plot(Markers1(f,:,1), Markers1(f,:,2),'ob')
    grid on
    axis equal
    axis(ejes)
    xlabel('X')
    ylabel('Y')
    title(['Frames: ' num2str(f)])
    drawnow
end

%adapted by Eduardo Saman
%contact: eduardos@uic.es
