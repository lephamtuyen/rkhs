function testRBFFeatures
global RBF S_dim A_dim S_max A_max S_min
X_train = [1:100]';
y_orig = sin(X_train/10) + (X_train/50).^2;
y_train = y_orig + (0.2 * randn(100, 1)); % Add noise

S_dim = 1; S_max = 100;S_min = 1;

% Define RBF
dimensions = [1];
partitions = [10];
initializeRBF(S_dim,dimensions,partitions);

% training
X_activ = zeros(100, RBF.numFeatures);
for i =1:100
    input = X_train(i,:);
    phi = getPhi(input);
    X_activ(i, :) = phi;
end
X_activ = [ones(100, 1), X_activ];
Theta = pinv(X_activ' * X_activ) * X_activ' * y_train;   

xs = [1:0.5:100]';
ys = zeros(size(xs));
for i = 1:length(xs)
	phis = getPhi(xs(i));
    phis = [1, phis];
    ys(i) = phis*Theta;
end

% ==================================
%         Plot Result
% ==================================

figure(1);
hold on; 

% Plot the original function as a black line.
plot(X_train, y_orig, 'k-');

% Plot the noisy data as blue dots.
plot(X_train, y_train, '.');

% Plot the approximated function as a red line.
plot(xs, ys, 'r-');

legend('Original', 'Noisy Samples', 'Approximated');
axis([0 100 -1 5]);
title('RBFN Regression');
end

function normalizeState = getNormalizeState(state)
global S_max S_min
normalizeState = (state-S_min)./(S_max-S_min);
end

function initializeRBF(dimens, dimensions, partitions)
global RBF S_max S_min

RBF.dimens = dimens;
RBF.dimensions = dimensions;
RBF.partitions = partitions;
RBF.numFeatures = prod(RBF.partitions);
RBF.sigmas = (S_max-S_min)./(2*RBF.partitions);
RBF.centers = linspace(RBF.sigmas(1),S_max(1)-RBF.sigmas(1),RBF.partitions(1));
for i=2:RBF.dimens
    temp = linspace(RBF.sigmas(i),S_max(i)-RBF.sigmas(i),RBF.partitions(i));
    RBF.centers = combvec(RBF.centers,temp);
end
end

function phi = getPhi(x)
global RBF
diffs = bsxfun(@minus, RBF.centers, x);
phi = exp(-0.5 * sum(diffs.^2 ./ repmat(RBF.sigmas.^2,1,RBF.numFeatures),1));
phi = phi/sum(phi);
end