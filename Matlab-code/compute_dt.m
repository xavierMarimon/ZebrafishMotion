function [tD_M1,tD_M2, tD_M3, speedH, speedT] = compute_dt(marker1,marker2, marker3, seconds)


%This function computes the distance traveled by each marker through the
%use of the known pythagorean theorem. THrough triangulation, the
%difference between point_x and point_x+1 is computed for the x and y axes,
%the sum of the change_y = sqrt(change_a+ change_b) is resolved. This is
%done foronly three markers, if more are required add the succession of
%each calculation (distance_marker, td_marker) for the markers necessary. 

distance_Marker1 = zeros(length(marker1), 1);
distance_Marker2 = zeros(length(marker2), 1);
distance_Marker3 = zeros(length(marker2), 1);

for i = 1:length(marker1)-1
    distance_Marker1(i,1) = hypot((marker1(i,1)-marker1(i+1,1)),(marker1(i,2)-marker1(i+1,2)));
    distance_Marker2(i,1) = hypot((marker2(i,1)-marker2(i+1,1)),(marker2(i,2)-marker2(i+1,2)));
    distance_Marker3(i,1) = hypot((marker3(i,1)-marker3(i+1,1)),(marker3(i,2)-marker3(i+1,2)));
end

tD_M1 = sum(distance_Marker1(:,1));
tD_M2 = sum(distance_Marker2(:,1));
tD_M3 = sum(distance_Marker3(:,1));

%extract the mean speec, the scalar value of the average velocity of the
%head and tail, only head is outputed!
speedH = tD_M1/seconds;
speedT = tD_M2/seconds;
end


%by Eduardo Saman
%contact:eduardos@uic.es