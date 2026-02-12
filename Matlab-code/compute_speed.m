function [velocity_M1, velocity_M2, velocity_M3, speed] = compute_speed(distance_Marker1,distance_Marker2, distance_Marker3, seconds)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

fps = round(length(distance_Marker1())/30);

i = 1:100:length(distance_Marker1);

velocity_M1 = zeros(length(i)-1,1);
velocity_M2 = zeros(length(i)-1,1);
velocity_M3 = zeros(length(i)-1,1);

for j = 1:(length(i)-1)
   velocity_M1(j,1) = sum(distance_Marker1(i(1,j):i(1,j)+fps,1));
   velocity_M2(j,1) = sum(distance_Marker2(i(1,j):i(1,j)+fps,1));
   velocity_M3(j,1) = sum(distance_Marker3(i(1,j):i(1,j)+fps,1));
end

speed = mean(velocity_M1);


end
