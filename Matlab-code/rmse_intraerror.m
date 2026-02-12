



%RMSE for calculating intra error
figure(1);
plot(Head(:,1), 'LineWidth', 2, 'Color', 'r');
hold on;
xlim([1 length(Head)]);
xlabel('Frame index');
ylabel('position (pixels)');
title('Interpolation between DLC RSx signal and MOTIV RSx signal');

for x = 1:length(Head)
    meanHead(x,1) = mean(Head(x,:));
end

for x = 1:length(Head)
    Head_RMSE(x,1) = sqrt(mean((meanHead(x,1)-Head(x,1))^2));
    Head_RMSE(x,2) = sqrt(mean((meanHead(x,1)-Head(x,2))^2));
    Head_RMSE(x,3) = sqrt(mean((meanHead(x,1)-Head(x,3))^2));
    Head_RMSE(x,4) = sqrt(mean((meanHead(x,1)-Head(x,4))^2));
    Head_RMSE(x,5) = sqrt(mean((meanHead(x,1)-Head(x,5))^2));
    Head_RMSE(x,6) = sqrt(mean((meanHead(x,1)-Head(x,6))^2));
    Head_RMSE(x,7) = sqrt(mean((meanHead(x,1)-Head(x,7))^2));
    Head_RMSE(x,8) = sqrt(mean((meanHead(x,1)-Head(x,8))^2));
    Head_RMSE(x,9) = sqrt(mean((meanHead(x,1)-Head(x,9))^2));
    Head_RMSE(x,10) = sqrt(mean((meanHead(x,1)-Head(x,10))^2));
end

%RMSE plot
figure(13);
plot(Head_RMSE(:,2), 'LineWidth', 2, 'Color', 'r');
hold on
plot(Head_RMSE(:,1), 'LineWidth', 2, 'Color', 'b');
hold on
plot(Head_RMSE(:,3), 'LineWidth', 2, 'Color', [0 50 80]);
grid on
xlabel('Frame index');
ylabel('RMSE');
legend('RMSE');
title('Root Mean Square Error for Right Shoulder X values');
hold off

