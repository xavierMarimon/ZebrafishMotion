
%Load data
%load('FeatureTable.mat');

%Acces table
%X=table2array(FeatureTable(:,2:20));

X = all(:,1:7)';
X = sickvshealthy(:,1:7)';
%% Center and standarize data
%returns the z-score along each row
Xnorm=normalize(X,2)';

%% Compute PCA
[coeff,score, latent]=pca(Xnorm);

%The sorted eigenvector (descending order) is known as coeff.
%The projected data is known as scores.
%The sorted eigenvalue (descending order) is known as latent. dispersion 

%% Plot
h1=figure(1);
axesActual=axes('Parent',h1);
set(axesActual,'FontSize',18);
set(gca,'FontSize',18);
hold on;
%Plot PCA space of the first two PCs: PC1 and PC2
sz=100;
s=scatter(score(:,1),score(:,2),sz);
s.LineWidth = 1;
s.MarkerFaceColor = 'k';
s.MarkerEdgeColor = 'k';
grid on;
xlabel('PC 1');
ylabel('PC 2');
title('Data projected');
%nicecolor;
hold off;

% saveas(figure(1), 'plot_pca_analysis', 'fig');
% saveas(figure(1), 'plot_pca_analysis', 'jpg');

%% Scree plot

%Percent of variance explained
varExplain=(latent/sum(latent))*100;

%cummulative percent of variance explained
cumvarExplain=cumsum(varExplain);

%% Plot
h2=figure(2);
axesActual=axes('Parent',h2);
set(axesActual,'FontSize',18);
set(gca,'FontSize',18);
hold on;
%Scree plot
h=plot(1:1:length(latent),latent);
set(h,'LineWidth',2 ,'Color','k','Marker','o', 'MarkerFaceColor','k',...
'MarkerEdgeColor','k','MarkerSize',4);
%Plot title
title('Scree plot');
%Labels
ylabel('eigenvalues');
xlabel('Principal component coefficients');
%Create XTickLabels
charTick= cell(length (latent),1);

for i=1:length(latent)
    charTick{i} = ['PC ' num2str(i)];
end
%Set XTickLabels to current axes
set(gca, 'XTick',1:length (latent), 'XTickLabel',charTick);
grid on;
%nicecolor;
hold off;

h3=figure(3);
axesActual=axes('Parent',h3);
set(axesActual,'FontSize',18);
set(gca,'FontSize',18);
hold on;
%Scree plot
bar(varExplain,'FaceColor','k','EdgeColor','k');
grid on;
h=plot(1:1:length(cumvarExplain),cumvarExplain);
set(h,'LineWidth',2 ,'Color','k','Marker','o', 'MarkerFaceColor','k',...
'MarkerEdgeColor','k','MarkerSize',4);
%Plot title
title('Scree plot');
%Labels
ylabel('Variance explained (%)');
xlabel('Principal component coefficients');
%Create XTickLabels
charTick= cell(length (latent),1);

for i=1:length(latent)
    charTick{i} = ['PC ' num2str(i)];
end
%Legend
legend('Variance explained', 'Cumulative variance explained', 'Location','best');
%Set XTickLabels to current axes
set(gca, 'XTick',1:length (latent), 'XTickLabel',charTick);
%nicecolor;
hold off;
% 
% saveas(figure(3), 'screeplot_pca_analysis', 'fig')
% saveas(figure(3), 'screeplot_pca_analysis', 'jpg')

idx_pca = kmeans([score(:,1), score(:,2)],2);
gscatter(score(:,1), score(:,2),idx_pca, "kr", ".", 25);
xlabel('PC1', 'FontSize',12);
ylabel('PC2', 'FontSize', 12);
legend('Cluster 1', 'Cluster 2', 'Location','best');
title('K-Means Classification with PCA', 'FontSize',14);
subtitle('C = 2 | PCs = 2', 'FontSize',10);
saveas(figure(1), 'kmeans_pca_analysis_kmeans', 'fig')
saveas(figure(1), 'kmeans_pca_analysiskmeans', 'jpg')

idx = kmeans([nor_sickhealthy(:,1), nor_sickhealthy(:,5)],2);
figure(1);
gscatter(nor_sickhealthy(:,1), nor_sickhealthy(:,5),idx, "mg", ".", 20);
xlabel('Normalized Distance Traveled', 'FontSize',12);
ylabel('Normalized Approximate Entropy', 'FontSize', 12);
legend('Pathological Fish', 'Healthy Fish', 'Location','northwest');
title('Pathological Fish vs. Healthy Fish K-Means Classification', 'FontSize',14);
subtitle('C = 2', 'FontSize',10);

% saveas(figure(1), 'Kmeans Healthy vs Pathological Classification', 'fig')
% saveas(figure(1), 'Kmeans Healthy vs Pathological Classification', 'jpg')






