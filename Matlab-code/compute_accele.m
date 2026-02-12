function [acceleration1,acceleration2, acceleration3] = compute_accele(a1,a2,a3)
%compute the acceleration automatically from all markers using the
%dlc2kinematics values (x,y)


acceleration1 = zeros(length(a1), 1);
acceleration2 = zeros(length(a2), 1);
acceleration3 = zeros(length(a3), 1);


for i = 1:length(acceleration1)
    acceleration1(i,1) = hypot(a1(i,1),a1(i,2));
    acceleration2(i,1) = hypot(a2(i,1),a2(i,2));
    acceleration3(i,1) = hypot(a3(i,1),a3(i,2));
end

end

%by Eduardo Saman
%contact: eduardos@uic.es