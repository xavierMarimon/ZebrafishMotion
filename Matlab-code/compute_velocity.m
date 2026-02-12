function [velocity1,velocity2, velocity3] = compute_velocity(v1,v2,v3)
%this function computes the velocity automatically for all markers


velocity1 = zeros(length(v1), 1);
velocity2 = zeros(length(v2), 1);
velocity3 = zeros(length(v3), 1);


for i = 1:length(velocity1)
    velocity1(i,1) = hypot(v1(i,1),v1(i,2));
    velocity2(i,1) = hypot(v2(i,1),v2(i,2));
    velocity3(i,1) = hypot(v3(i,1),v3(i,2));
end

end

%by Eduardo Saman
%contact:eduardos@uic.es