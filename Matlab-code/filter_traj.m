function [m1_Filtered,m2_Filtered,m3_Filtered] = filter_traj(m1,m2,m3, seconds, filter_value)
%This function automatically applies a filter to all trajectory signals
%Apply a lowpass filter of (filter_value) to the values, to smooth the signal. 

mm1_Filtered = lowpass(m1, filter_value, (length(m1)/seconds));
mm2_Filtered = lowpass(m2,filter_value, (length(m2)/seconds));
mm3_Filtered = lowpass(m3,filter_value, (length(m3)/seconds));


m1_Filtered = mm1_Filtered(10:(end-50),1:2);
m2_Filtered = mm2_Filtered(10:(end-50),1:2);
m3_Filtered = mm3_Filtered(10:(end-50),1:2);
end