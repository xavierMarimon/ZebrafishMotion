
clear all;
close all;
clc;

%DLC PLOTTING SCRIPT

for j = 5
    for i = 6
        %EXTRACT THE THREE MARKERS FROM THE .csv file
        [Head, Tail_B, Tail_E] = extract_marker_data(['Data/f' num2str(j) '/f' num2str(j) 'v' num2str(i) '/df' num2str(j) 's3_v' num2str(i) 'DLC_resnet50_ZebrafishNet20Apr28shuffle1_25000_filtered']);
        [VHead, VTail_B, VTail_E] = extract_marker_data(['Data/f' num2str(j) '/f' num2str(j) 'v' num2str(i) '/velocity']);
        [AHead, ATail_B, ATail_E] = extract_marker_data(['Data/f' num2str(j) '/f' num2str(j) 'v' num2str(i) '/acceleration']);
        
        %DEFINE PIXEL CONVERSION ONE TIME!!!
        % imtool(imread('image1.jpg'))
        %can be redefined with imtool() function
        px_in_cm = 22.3902;
        seconds = 30;
    
        %UNIT CONVERSION
        [Head_cm, Tail_B_cm, Tail_E_cm] = position_cm(Head, Tail_B, Tail_E, px_in_cm);
    
%%% SIGNAL SMOOTHING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [Head_Filtered,Tail_B_Filtered,Tail_E_Filtered] = filter_traj(Head_cm, Tail_B_cm, Tail_E_cm, seconds, 0.6);

        [VH_Filtered,VTB_Filtered,VTE_Filtered] = filter_traj(VHead,VTail_B,VTail_E, seconds, 0.6);
        [AH_Filtered,ATB_Filtered,ATE_Filtered] = filter_traj(AHead,ATail_B,ATail_E, seconds, 0.6);

%%% FEATURE EXTRACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %COMPUTE TOTAL DISTANCE TRAVELED, ONLY THE HEAD IS OUTPUTED AS ITS
        %THE MOST STABLE AND REPRESENTATIVE
        [tD_M1,tD_M2, tD_M3, speedH, speedT] = compute_dt(Head_Filtered, Tail_B_Filtered, Tail_E_Filtered, seconds);
        
        %COMPUTE VELOCITY AND MAXIMUM VELOCITY
        [vel_H, vel_TB, vel_TE] = compute_velocity(VH_Filtered, VTB_Filtered, VTE_Filtered);
        H_MaxVel = max(vel_H);
        T_MaxVel = max(vel_TB);

        %COMPUTE ACCELERATION AND MAXIMUM ACCELERATION
        [acc_H, acc_TB, acc_TE] = compute_accele(AH_Filtered, ATB_Filtered, ATE_Filtered);
        H_MaxAcc = max(acc_H);
        T_MaxAcc = max(acc_TB);

        %EXTRACT THE CHAOTIC FEATURES FROM HEAD AND TAILB
        H_appEnt = approximateEntropy(Head_Filtered);
        H_CD = correlationDimension(Head_Filtered);
        H_lyapExp = lyapunovExponent(Head_Filtered,(length(Head_Filtered)/seconds));
    
        T_appEnt = approximateEntropy(Tail_B_Filtered);
        T_CD = correlationDimension(Tail_B_Filtered);
        T_lyapExp = lyapunovExponent(Tail_B_Filtered,(length(Tail_B_Filtered)/seconds));

        %SAVE RESULTS IN A NUMERIC ARRAY
        results_Head = [tD_M1, speedH, H_MaxVel, H_MaxAcc, H_appEnt, H_CD, H_lyapExp];
        results_Tail = [tD_M2, speedT, T_MaxVel, T_MaxAcc, T_appEnt, T_CD, T_lyapExp];
        
        %EXPORT MATRIX TO CORRESPONDING VIDEO DIRECTORY
        writematrix(results_Head, ['Data/f' num2str(j) '/f' num2str(j) 'v' num2str(i) '/f' num2str(j) 'v' num2str(i) '_Head_Results.csv']);
        writematrix(results_Tail,['Data/f' num2str(j) '/f' num2str(j) 'v' num2str(i) '/f' num2str(j) 'v' num2str(i) '_Tail_Results.csv']);

%%% DYNAMIC MARKER ILLUSTRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %This section is special, it can be used or not, but its a
        %dynamic representation of the fish motion
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

%%% SIGNAL BEFORE FILTERING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %THIS FIGURE SHOWS THE MARKER BEFORE THE FILTERING
        %figure(1)
        % plot(Head_cm(:,1), Head_cm(:,2), "LineStyle","-", "LineWidth",2, "Color",'b');
        % hold on
        % plot(Tail_B_cm(:,1), Tail_B_cm(:,2),"LineStyle","-", "LineWidth",2, "Color",'r');
        % %plot(Tail_E(:,1), Tail_E(:,2),"LineStyle","-", "LineWidth",1, "Color",'k');
        % hold off;
        % grid on;
        % legend("Head", 'TailB', 'TailE');
        % xlabel('Tank Position X (cm)', 'FontSize',14);
        % ylabel('Tank Position Y (cm)', 'FontSize',14);


%%% SIGNAL PLOTTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        figure(1);
        plot(Head_cm(:,1), Head_cm(:,2), "LineStyle","-", "LineWidth",2, "Color",'b', 'Marker', '>', 'MarkerIndices', 1,'MarkerEdgeColor','k', 'MarkerFaceColor','b', 'MarkerSize',10);
        hold on;
        plot(Head_cm(:,1), Head_cm(:,2), "LineStyle","-", "LineWidth",2, "Color",'b','Marker', 'o','MarkerIndices', length(Head_Filtered),'MarkerEdgeColor','k', 'MarkerFaceColor','b', 'MarkerSize',10);
        hold on;
        plot(Tail_B_cm(:,1), Tail_B_cm(:,2),"LineStyle","-", "LineWidth",2, "Color",'r', 'Marker', '>', 'MarkerIndices', 1, 'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10);
        hold on;
        plot(Tail_B_cm(:,1), Tail_B_cm(:,2),"LineStyle","-", "LineWidth",2, "Color",'r', 'Marker', 'o', 'MarkerIndices', length(Tail_B_Filtered), 'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10);
        %hold on
        %plot(Tail_E_Filtered(:,1), Tail_E_Filtered(:,2),"LineStyle","-", "LineWidth",2, "Color",'g', 'Marker', '>', 'MarkerIndices', 1, 'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10);
        %hold on;
        %plot(Tail_E_Filtered(:,1), Tail_E_Filtered(:,2),"LineStyle","-", "LineWidth",2, "Color",'g', 'Marker', 'o', 'MarkerIndices', length(Tail_B_Filtered), 'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10);
        hold off;
        grid on;
        xlim([-1 21]);
        ylim([-1 21]);
        legend("Head Start","Head End", 'TailB Start','TailB End', 'Location','best');
        xlabel('Tank Position X (cm)', 'FontSize',12);
        ylabel('Tank Position Y (cm)', 'FontSize',12);
        title(['Trajectory f' num2str(j) ' v' num2str(i)], "FontSize",14);
        txt = ['DT = ' num2str(tD_M1) ' cm |', ' Speed = ' num2str(speedH) ' cm/s |', ' MaxV = ' num2str(H_MaxVel) ' cm/s |', ' MaxA = ' num2str(H_MaxAcc) ' cm/s^2'];
        subtitle(txt, "FontSize",8.5);
    
        saveas(figure(1),['Data/f' num2str(j) '/f' num2str(j) 'v' num2str(i) '/Trajectory f' num2str(j) ' v' num2str(i)], 'fig');
        saveas(figure(1),['Data/f' num2str(j) '/f' num2str(j) 'v' num2str(i) '/Trajectory f' num2str(j) ' v' num2str(i)], 'jpg');
    
        close all;
        clearvars -except j
    end

end


%by Eduardo Saman
%contact: eduardos@uic.es


