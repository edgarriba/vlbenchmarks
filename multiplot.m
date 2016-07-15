dataset_dir = '/home/eriba/datasets/';

dataset = 'fountain_dense';
%dataset = 'herzjesu_dense';

nn_method = 'nn';
nn_method = 'nnr';

detector = 'harris';
%detector = 'dog';

solver = 'stewenius';
% solver = 'nister';
% solver = 'sevenpt';
% solver = 'eightpt';

test = 'r';
% test = 't';
 test = 'matches';
 test = 'inliers';
 test = 'inliers_nn';
% test = 'inliers_all';
 test = 'iterations';
 test = 'timing';
 test = 'rep_error';

fname = strcat(dataset_dir, dataset, '/urd/results/',...
               nn_method, '_', detector, '_', solver, '_', test, '.txt');
           
M = load(fname);
M(isnan(M))=0;
M(isinf(M))=0;

N = 3;
num_experiments = size(M, 1) / N;
num_sequences   = size(M, 2);

A = M(0*num_experiments+1:1*num_experiments, :);
B = M(1*num_experiments+1:2*num_experiments, :);
C = M(2*num_experiments+1:3*num_experiments, :);

GroupedData = {A B C};

legendEntries = {'IRI' 'SIFT' 'TFeat'};

x = GroupedData;

figure;
h = boxplot2(permute(cat(N, x{:}), [2 3 1]), 1:num_sequences);

legend(legendEntries)
ylabel(upper(test))
xlabel('Sequences')
title(strcat(dataset,'-',nn_method,'-',detector,'-',solver))