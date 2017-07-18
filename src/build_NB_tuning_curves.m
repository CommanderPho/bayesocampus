function [coords,alpha,beta] = build_NB_tuning_curves(...
    spikes,X,t,sample_rate,t_start,t_end,bin_size,sigma,f_base,min_t_occ)
%{
Build tuning curves from spiking data via a sequential Bayesian estimate of
a distribution over firing rate (using a gamma-distributed prior).

	Args:
		spikes: 1 x K cell array of spiking data, where K is the number
        of units:
			spikes{1}: vector of timestamps for each spike at unit 1.
			spikes{2}: vector of timestamps for each spike at unit 2.
			...
		X: N x M array of ground-truth stimulus features, where N is the
        number of stimulus dimensions and M is the number of sampling
        timestamps.
		t: 1 x M array of timestamps for ground-truth stimulus sampling.
        sample_rate: sample rate of ground-truth data.
		t_start: vector of start times for sampling windows over which to
        construct tuning curves.
		t_end: vector of end times for sampling windows over which to
        construct tuning curves (must be same length as t_start).
		bin_size: length-N vector containing the bin size to use for each
        stimulus dimension.
		sigma: length-N vector containing the bandwidth along each
        dimension for the Gaussian kernel used to smooth occupancy and
        spike counts.
        f_base: (scalar) expected basal firing rate for a neuron.
        min_t_occ: (scalar) expected minimum sampling time at each
        location.

    Returns:
        coords: N x 1 cell array of coordinate locations, where N is the
        number of stimulus dimensions:
            coords{1}: N-dimensional array of the coordinate position for
            each bin in stimulus dimension 1.
            coords{2}: N-dimensional array of the coordinate position for
            each bin in stimulus dimension 2.
            ...
        alpha: (N+1)-dimensional array of alpha parameter for each bin,
        where the size of the first dimension is the number of units.
        beta: N-dimensional array of beta parameter for each bin.
%}

% Only use ground-truth data from the specified intervals.
t_use = get_interval_logical(t,t_start,t_end);

X = X(:,t_use);
t = t(t_use);

% Put the ground-truth stimulus data into bins.
N = size(X,1);  % Dimensionality of stimulus data.
X_b = zeros(size(X));
grid_vectors = cell(1,N); % Coordinates in dimension N for each bin.
occ_size = zeros(1,N);  % The size (in bins) of the occupancy space.
for i = 1:N
    % Get the locations of every bin along this dimension.
    bins = min(X(i,:)):bin_size(i):max(X(i,:)) + bin_size(i)/2;
    grid_vectors{i} = bins;
    occ_size(i) = length(grid_vectors{i});
    
    bin_indices = 1:occ_size(i);
    
    X_b(i,:) = interp1(bins,bin_indices,X(i,:),'nearest');
end

% Get coordinates for each bin.
coords = cell(N,1);
[coords{:}] = ndgrid(grid_vectors{:});

% Get occupancy.
t_occ = get_bin_counts(X_b,occ_size)*sample_rate;

empties = t_occ == 0;

% Smooth occupancy by convolving with a Gaussian kernel.
G = build_kernel(bin_size,sigma);
t_occ = convn(t_occ,G,'same');

t_occ(empties) = nan;

% Make a Bayesian estimate of the parameters for a gamma distribution over
% firing rate at each point in occupancy space.
alpha_0 = f_base*min_t_occ; % Minimum number of spikes expected.
beta_0 = min_t_occ; % Minimum number of unit temporal bins expected.

beta = beta_0 + t_occ;  % Compute beta parameter.

if N == 1
    assign_mask = true(occ_size,1);
else
    assign_mask = true(occ_size);
end

K = length(spikes); % Number of units.
alpha = zeros([K occ_size]);
for i = 1:K
    ts = spikes{i};
    
    % Only use spike data from the specified intervals.
    ts_use = get_interval_logical(ts,t_start,t_end);
    ts = ts(ts_use);
    
    % Put the spiking data into bins in stimulus space.
    ts_b = interp1(t,X_b',ts,'nearest')';
    
    % Get spike counts at each location.
    n = get_bin_counts(ts_b,occ_size,empties);
    
    % Smooth spike counts.
    n = convn(n,G,'same');
    
    alpha(i,assign_mask) = alpha_0 + n(:);   % Compute alpha parameter.
end

end