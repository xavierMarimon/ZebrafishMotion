
clear all;
close all;
clc;

for j = 2:3
    for i = 1:10
        %Extract only two markers from the .csv file
        [Head, Tail_B, Tail_E] = extract_marker_data(['Data/sf' num2str(j) '/sf' num2str(j) 'v' num2str(i) '/dsf' num2str(j) 's5_v' num2str(i) 'DLC_resnet50_ZebrafishNet20Apr28shuffle1_25000_filtered']);
        [VHead, VTail_B, VTail_E] = extract_marker_data(['Data/sf' num2str(j) '/sf' num2str(j) 'v' num2str(i) '/velocity']);
        [AHead, ATail_B, ATail_E] = extract_marker_data(['Data/sf' num2str(j) '/sf' num2str(j) 'v' num2str(i) '/acceleration']);
    
    % imtool(imread('image1.jpg'))
    
    
    %can be redefined with imtool() function
        px_in_cm = 22.3902;
        seconds = 30;
    
        [Head_cm, Tail_B_cm, Tail_E_cm] = position_cm(Head, Tail_B, Tail_E, px_in_cm);
    
        [Head_Filtered,Tail_B_Filtered,Tail_E_Filtered] = filter_traj(Head_cm, Tail_B_cm, Tail_E_cm, seconds);
        [VH_Filtered,VTB_Filtered,VTE_Filtered] = filter_traj(VHead,VTail_B,VTail_E, seconds);
        [AH_Filtered,ATB_Filtered,ATE_Filtered] = filter_traj(AHead,ATail_B,ATail_E, seconds); 
    
    
        [tD_M1,tD_M2, tD_M3, speedH, speedT] = compute_dt(Head_Filtered, Tail_B_Filtered, Tail_E_Filtered, seconds);
        [vel_H, vel_TB, vel_TE] = compute_velocity(VH_Filtered, VTB_Filtered, VTE_Filtered);
        [acc_H, acc_TB, acc_TE] = compute_accele(AH_Filtered, ATB_Filtered, ATE_Filtered);
    
        H_MaxVel = max(vel_H);
        H_MaxAcc = max(acc_H);
    
        T_MaxVel = max(vel_TB);
        T_MaxAcc = max(acc_TB);
    
    
    
    %[vH, vTB, vTE, speedH] = compute_speed(distance_Marker1, distance_Marker2, distance_Marker3, seconds);
    % [accelH, accelTB, accelTE] = compute_acceleration(vH,vTB,vTE);
    
    % %Data = dlc2strc(Marker1_Data, 'Name of Marker1', Marker2_Data, 'Name of Marker2');
    % Data = dlc2strc(Head_cm, 'Head', Tail_B_cm, 'Tail_Base', Tail_E_cm, 'Tail_End');
    % 
    % %Plot
    % figure(1);
    % Speed=8;
    % plotmarkers(Data,Speed);
    % 
    % Data_Filtered = dlc2strc(Head_Filtered, 'Head', Tail_B_Filtered, 'Tail_Base', Tail_E_Filtered, 'Tail_End');
    % %Plot
    % figure(1);
    % Speed=8;
    % plotmarkers(Data_Filtered,Speed);
    
    
    
        figure(1);
    % subplot(1,2,1)
    % plot(Head_cm(:,1), Head_cm(:,2), "LineStyle","-", "LineWidth",2, "Color",'b');
    % hold on
    % plot(Tail_B_cm(:,1), Tail_B_cm(:,2),"LineStyle","-", "LineWidth",2, "Color",'r');
    % %plot(Tail_E(:,1), Tail_E(:,2),"LineStyle","-", "LineWidth",1, "Color",'k');
    % hold off;
    % grid on;
    % legend("Head", 'TailB', 'TailE');
    % xlabel('Tank Position X (cm)', 'FontSize',14);
    % ylabel('Tank Position Y (cm)', 'FontSize',14);
    
    % subplot(1,2,2)
        plot(Head_Filtered(:,1), Head_Filtered(:,2), "LineStyle","-", "LineWidth",2, "Color",'b', 'Marker', '>', 'MarkerIndices', 1,'MarkerEdgeColor','k', 'MarkerFaceColor','b', 'MarkerSize',10);
        hold on;
        plot(Head_Filtered(:,1), Head_Filtered(:,2), "LineStyle","-", "LineWidth",2, "Color",'b','Marker', 'o','MarkerIndices', length(Head_Filtered),'MarkerEdgeColor','k', 'MarkerFaceColor','b', 'MarkerSize',10);
        hold on;
        plot(Tail_B_Filtered(:,1), Tail_B_Filtered(:,2),"LineStyle","-", "LineWidth",2, "Color",'r', 'Marker', '>', 'MarkerIndices', 1, 'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10);
        hold on;
        plot(Tail_B_Filtered(:,1), Tail_B_Filtered(:,2),"LineStyle","-", "LineWidth",2, "Color",'r', 'Marker', 'o', 'MarkerIndices', length(Tail_B_Filtered), 'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10);
    % plot(Tail_E_Filtered(:,1), Tail_E_Filtered(:,2),"LineStyle","-", "LineWidth",1, "Color",'g');
        hold off;
        grid on;
        xlim([-1 21]);
        ylim([-1 21]);
        legend("Head Start","Head End", 'TailB Start','TailB End', 'Location','best');
        xlabel('Tank Position X (cm)', 'FontSize',12);
        ylabel('Tank Position Y (cm)', 'FontSize',12);
        title(['Trajectory sf' num2str(j) ' v' num2str(i)], "FontSize",14);
        txt = ['DT = ' num2str(tD_M1) ' cm |', ' Speed = ' num2str(speedH) ' cm/s |', ' MaxV = ' num2str(H_MaxVel) ' cm/s |', ' MaxA = ' num2str(H_MaxAcc) ' cm/s^2'];
        subtitle(txt, "FontSize",8.5);
    
    
        saveas(figure(1),['Data/sf' num2str(j) '/sf' num2str(j) 'v' num2str(i) '/Trajectory sf' num2str(j) ' v' num2str(i)], 'fig');
        saveas(figure(1),['Data/sf' num2str(j) '/sf' num2str(j) 'v' num2str(i) '/Trajectory sf' num2str(j) ' v' num2str(i)], 'jpg');
    % % sgtitle('f1 21 Control Before and After Filter');
    % % 
    % figure(2)
    % plot(vel_H, '-b', 'LineWidth',2);
    % hold on;
    % plot(vel_TB, '-r', 'LineWidth',2);
    % hold on;
    % % plot(vel_TE, '-g', 'LineWidth',2);
    % hold on;
    % grid on;
    % legend("Head", 'TailB', 'Location','best');
    % xlabel('Time (Seconds)', 'FontSize',14);
    % ylabel('Velocity (cm/s)', 'FontSize',14);
    % title('Velocity');
    % hold off;
    
    % saveas(figure(2),'RESULTS/f4/f4v8/Velocity', 'jpg');
    % %  
    % figure(3)
    % plot(acc_H, '-b', 'LineWidth',2);
    % hold on;
    % plot(acc_TB, '-r', 'LineWidth',2);
    % hold on;
    % %plot(acc_TE, '-g', 'LineWidth',2);
    % hold on;
    % grid on
    % legend("Head", 'TailB', 'Location','best');
    % xlabel('Time (Seconds)', 'FontSize',14);
    % ylabel('Acceleratoin (cm/s^2)', 'FontSize',14);
    % title('Acceleration');
    % hold off;
    % saveas(figure(2),'RESULTS/f4/f4v8/Acceleration', 'jpg');
    
        H_appEnt = approximateEntropy(Head_Filtered);
        H_CD = correlationDimension(Head_Filtered);
        H_lyapExp = lyapunovExponent(Head_Filtered,(length(Head_Filtered)/seconds));
    
        T_appEnt = approximateEntropy(Tail_B_Filtered);
        T_CD = correlationDimension(Tail_B_Filtered);
        T_lyapExp = lyapunovExponent(Tail_B_Filtered,(length(Tail_B_Filtered)/seconds));
    
        results_Head = [tD_M1, speedH, H_MaxVel, H_MaxAcc, H_appEnt, H_CD, H_lyapExp];
    
        results_Tail = [tD_M2, speedT, T_MaxVel, T_MaxAcc, T_appEnt, T_CD, T_lyapExp];
    
        writematrix(results_Head, ['Data/sf' num2str(j) '/sf' num2str(j) 'v' num2str(i) '/sf' num2str(j) 'v' num2str(i) '_Head_Results.csv']);
        writematrix(results_Tail,['Data/sf' num2str(j) '/sf' num2str(j) 'v' num2str(i) '/sf' num2str(j) 'v' num2str(i) '_Tail_Results.csv']);
        close all;
        clearvars -except j
    end

end





