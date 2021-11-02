%% Set things up.
clear
addpath('../src')

%% Load the data.
file_path = '../../EichenbaumData/AJF023/EF3/AJF023EF3SpksEvs.mat';
[spikes, X, t, sample_rate] = load_data_xy(file_path);
% how to spell "TURD"?
%{
spikes: 1xK cell array of spiking data, where K is the number of units:
            spikes{1}: vector of timestamps for each spike at unit 1.
            spikes{2}: vector of timestamps for each spike at unit 2.
            ...
        X: NxM array of ground-truth stimulus features, where N is the number of
        stimulus dimensions and M is the number of sampling timestamps.
        t: 1xM array of timestamps for ground-truth stimulus sampling.
        sample_rate: sample rate of ground-truth data.
%}

spikes = source_data.spikes.RoyMaze1.time;
% Absolute timestamp version are available here: source_data.position.RoyMaze1.t

X = spikeStruct.linearPos;
t = spikeStruct.t;

sample_rate = 1/source_data.basics.RoyMaze1.posSampleRate; % 1 / 29.970;


%% Pho's comments:
% Looks like I need some position data in X


%X = X(1,:); % For testing on one-dimensional data.
%X = [X; X(1,:)]; % For testing on three-dimensional data.

%% Build the tuning curves.
t_start = t(1);
t_end = t(end);
sigma = [3 3];
bin_size = [3 3];
f_base = 2;
min_t_occ = 0.5;

[~,lambda] = build_tuning_curves(spikes,X,t,sample_rate,t_start,t_end,bin_size,sigma);
[coords,alpha,beta] = build_tuning_curves(spikes,X,t,sample_rate,t_start,t_end,bin_size,sigma,f_base,min_t_occ);

%% Get information content curves.
IC_curves = get_IC_curves(alpha,beta,f_base,min_t_occ);

%% Page through the tuning curves and information content curves.
K = size(alpha,1);
for i = 1:K
    subplot(3,1,1);
    plot_curves('tuning_curve',coords,lambda(i,:,:));
    title(sprintf('Poisson-based tuning curve %d of %d',i,K));
    subplot(3,1,2);
    plot_curves('tuning_curve',coords,alpha(i,:,:),beta);
    title(sprintf('Negative binomial-based tuning curve %d of %d',i,K));
    subplot(3,1,3);
    plot_curves('IC_curve',coords,IC_curves(i,:,:));
    title(sprintf('Information content curve %d of %d',i,K));
    
    waitforbuttonpress;
end

%% Perform neural decoding.
t_0 = 1000;
t_f = 1030;
t_step = 0.25;
t_start = t_0:t_step:t_f;
t_end = (t_0:t_step:t_f) + t_step;
 
poiss_posterior = bayesian_decode(spikes,t_start,t_end,lambda);
nb_posterior = bayesian_decode(spikes,t_start,t_end,alpha,beta);

%% Page through the decoded posterior across time.
M = size(nb_posterior,1);
for i = 1:M
    t = t_start(i);
    
    subplot(2,1,1);
    plot_curves('posterior',coords,poiss_posterior(i,:,:));
    title(sprintf('Poisson-based decoded posterior at t = %.2f',t));
    subplot(2,1,2);
    plot_curves('posterior',coords,nb_posterior(i,:,:));
    title(sprintf('Negative binomial-based decoded posterior at t = %.2f',t));
    
    waitforbuttonpress;
end