clear
close all
clc
%Load data
regressed = readmatrix('full_cv_log_cv_test.csv');
ground_truth = readmatrix("train_gt.csv");

mse_baseline = [870.02899169921875, 3828.40234375, 1588.857421875, 0.0677162706851959228515625];

ds_cardinality = length(ground_truth);
soil_parameters_scores = zeros(ds_cardinality,4);
mean_scores = zeros(ds_cardinality,1);
diag_size = zeros(ds_cardinality,1);

for i=1:length(regressed)
    %Compute per-sample regression error score
    soil_parameters_scores(i,:) = ((regressed(i,2:5)-ground_truth(i,2:5)).^2)./mse_baseline;
    mean_scores(i) = mean(soil_parameters_scores(i,:));
    
    %Retrieve patch size
    diag_size(i) = sqrt(regressed(i,6)^2+regressed(i,7)^2);
end

%% Plot patch diagonal size histogram vs mean and median errors
nbins = 10;
bin_width = floor(max(diag_size)/nbins);

figure
grid on
yyaxis right
h=histogram(diag_size,nbins);
ylabel('Frequency','FontSize', 12)

mean_scores_partitioned_by_diag = zeros(nbins,1);
median_scores_partitioned_by_diag = zeros(nbins,1);

diag_bins = h.BinEdges;

for j = 1:length(diag_bins)-1
   mean_scores_partitioned_by_diag(j) = ...
       mean(mean_scores((diag_size>=diag_bins(j) & diag_size<diag_bins(j+1))));
   median_scores_partitioned_by_diag(j) = ...
       median(mean_scores((diag_size>=diag_bins(j) & diag_size<diag_bins(j+1))));
end

%Avoid NaN values (in case some diagonal sizes are not present in the dataset)
idxs = ~isnan(mean_scores_partitioned_by_diag);

%Find center of bins
x = (h.BinEdges(1:end-1)+h.BinEdges(2:end))/2;

yyaxis left
plot(x(idxs),mean_scores_partitioned_by_diag(idxs),'-ok')
hold on
plot(x(idxs),median_scores_partitioned_by_diag(idxs),'-ob')
ylabel('Competition metric','FontSize', 12)
xlabel('Patch diagonal size [px]','FontSize', 12)
legend({'Mean values','Median values'},'FontSize',12)
xticks(x)

%% Analyse worst results
%Highest mean errors
how_many = 3;

[max_k_errors, indices] = maxk(mean_scores,how_many);

for qq=1:how_many
    fprintf('\nWorst patch %i diag size: %.2f px\n',qq,diag_size(indices(qq)))
    fprintf('Worst patch %i P/K/Mg/Ph: %.2f \t %.2f \t %.2f \t%.2f\n',...
                                  qq,soil_parameters_scores(indices(qq),:))
    fprintf('Worst patch %i mean regression error: %.4f\n',...
                                             qq,mean_scores(indices(qq,:)))
end

max_gt_values = [325, 625, 400, 14];

header = {'Image id', 'P/Pmax', 'perc.','K/Kmax', 'perc.', 'Mg/Mgmax',...
    'perc.', 'Ph/Phmax', 'perc.'};

fprintf(['\nThe table below shows the ratio between each ground truth value and\n',...
    'the maximum of that parameter in the dataset followed by the percentile\n',...
    'it belogs to.\n'])
fprintf('\n%10s %10s %10s %10s %10s %10s %10s %10s %10s \n',header{:})
for i=1:length(indices)
    fprintf('%10.0i %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f \n',...
      regressed(indices(i),1),...
      ground_truth(indices(i),2)/max_gt_values(1),...
      sum(ground_truth(:,2)<=ground_truth(indices(i),2))/ds_cardinality*100,...
      ...
      ground_truth(indices(i),3)/max_gt_values(2),...
      sum(ground_truth(:,3)<=ground_truth(indices(i),3))/ds_cardinality*100,...
      ...
      ground_truth(indices(i),4)/max_gt_values(3),...
      sum(ground_truth(:,4)<=ground_truth(indices(i),4))/ds_cardinality*100,...
      ...
      ground_truth(indices(i),5)/max_gt_values(4),...
      sum(ground_truth(:,5)<=ground_truth(indices(i),5))/ds_cardinality*100)
end

fprintf('\nMean regression error: %.4f \n',mean(mean_scores))
fprintf('Median regression error: %.4f \n',median(mean_scores))