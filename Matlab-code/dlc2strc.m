function [Data_Struc] = dlc2strc(marker1_data, marker1name, marker2_data, maker2name, marker3_data, marker3name)

%this function converts the data from the markers to a structure.
%useful for the plotmarker function
seconds = 30;
[Frames markers]= size(marker1_data);
FR = (Frames/seconds);
time = zeros(1, Frames);


for x = 1:length(time)
    time(1,1) = 0;
    time(1, x+1) = (x*0.01);
end


Data_Struc = struct('Sets', struct('Fish', struct('Raw', struct(marker1name, marker1_data, maker2name, marker2_data, marker3name, marker3_data), 'NMarkers', 3)), 'Framerate', FR, 'Time', time, 'Frames', Frames);

end

%by Eduardo Saman
%contact: eduardos@uic.es