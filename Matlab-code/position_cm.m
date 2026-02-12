function [M1_cm, M2_cm, M3_cm] = position_cm(m1,m2,m3, px_in_cm)

%You must input the three markers (in our case we use three, modify and add
%variables in you added more. You also have to input the px in cm ratio,
%use the imtool function and an image depicting a line of 1cm to calculate
%it. 


%This function converts the position extracted from DLC in px to cm
%The first step is to create a matrix of 0 with the corresponding marker
%matrix size, to then automatically 

M1_cm = zeros(length(m1),2);
M2_cm = zeros(length(m2),2);
M3_cm = zeros(length(m3),2);

for i = 1:length(m1)
    M1_cm(i,1) = (m1(i,1)/px_in_cm);
    M1_cm(i,2) = (m1(i,2)/px_in_cm);
    M2_cm(i,1) = (m2(i,1)/px_in_cm);
    M2_cm(i,2) = (m2(i,2)/px_in_cm);
    M3_cm(i,1) = (m3(i,1)/px_in_cm);
    M3_cm(i,2) = (m3(i,2)/px_in_cm);
end

end

%Function by Eduardo Saman