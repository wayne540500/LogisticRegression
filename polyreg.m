% % loading dataset
% clc;
% clear;
% load dataset4.mat;
% 
% shapeX = size(X);
% % Declaration of parameter vector
% theta = ones(shapeX(2), 1); 
% theta_prev = zeros(shapeX(2), 1);
% 
% % number of iterations
% iteration = 1; 
% 
% % Learning rate
% alpha = 0.1;
% 
% % Max iterations
% max_iteration = 10000;
% 
% % R emperical vector
% costs = zeros(max_iteration, 1); 
% err = zeros(max_iteration, 1);
% 
% % Tolerence
% tolerance = 0.001;
% while (norm(theta - theta_prev) > tolerance) && (iteration < max_iteration)
% 
% [cost, gradient, f] = Remp(X, Y, theta);
% 
% theta_prev = theta;
% theta = theta - alpha * gradient;
% 
% costs(iteration) = cost;
% 
% [err_temp] = Prediction(X, Y, theta);
% 
% err(iteration) = err_temp;
% 
% iteration = iteration + 1;
% end
% 
% 
% disp("Number of iterations");
% disp(iteration - 1);
% 
% 
% subplot(2,1,1)
% plot(1 : iteration - 1, costs(1 : iteration - 1))
% title("Empirical Risk")
% 
% subplot(2,1,2)
% plot(1 : iteration - 1, err(1 : iteration - 1))
% title("Binary Classification Error")
% figure()
% 
% 
% X_out = X(Y == 0, :, :);
% disp(X(1, 1))
% 
% X_out1 = X(Y == 1, :, :);
% XX = (-theta(3)-X(:, 1) * theta(1)) / theta(2);
% 
% 
% %gscatter(X(:,1),X(:,2),Y,’br’,’xo’) requires machine learning toolbox
% scatter(X_out(:, 1),X_out(:, 2),'g')
% hold on
% 
% scatter(X_out1(:, 1),X_out1(:, 2),'r')
% hold on
% 
% plot(X(:, 1),XX,'k')
% title("Decision Boundary");
% legend('0','1',"Linear Decision Boundary")
% 
% 
% 
% % Caculate Remp and graduent
% function [cost, gradient, f] = Remp(X, Y, theta)
% 
% gradient = zeros(size(theta));
% 
% f = 1./(1 + exp(-theta'*X'))';
% 
% cost = (-1/length(Y)) * sum(Y.*log(f) + (1 - Y).*log(1 - f));
% 
% for j = 1 : size(gradient)
% gradient(j) = (1/length(Y)) * sum((f - Y).*X(:, j));
% end
% end
% 
% 
% 
% % Prediction using the computed theta
% 
% function [error] = Prediction(X, Y, theta)
% 
% f = 1./(1 + exp(-theta'*X'))';
% 
% error = 0;
% 
% for i = 1 : size(X)
% if(f(i) >= 0.5)
% f(i) = 1;
% end
% if(f(i) < 0.5)
% f(i) = 0;
% end
% end
% err = Y - f;
% for i = 1 : size(X)
% if(err(i) ~= 0)
% error = error + 1;
% end
% end
% end

%chatgpt
% Load the dataset from 'dataset4.mat'
clc;
clear;
load dataset4.mat;

% Define hyperparameters
alpha = 0.01;        % Learning rate
maxIterations = 10000;
tolerance = 0.0001;

% Initialize parameters
[m, n] = size(X);   % m: number of samples, n: number of features
theta = ones(n, 1);
previousTheta = zeros(n, 1);

% Initialize arrays to store errors and empirical risk
costHistory = zeros(maxIterations, 1);
classificationErrorHistory = zeros(maxIterations, 1);

% Gradient Descent
for iteration = 1:maxIterations
    % Compute logistic hypothesis
    hypothesis = 1./(1 + exp(-X * theta));
    
    % Calculate the gradient of the logistic loss
    gradient = (X' * (hypothesis - Y)) / m;
    
    % Update theta using gradient descent
    theta = theta - alpha * gradient;
    
    % Compute binary classification error
    predictions = (hypothesis >= 0.5);
    classificationError = sum(predictions ~= Y) / m;
    
    % Compute empirical risk (logistic loss)
    cost = (-1/m) * sum(Y .* log(hypothesis) + (1 - Y) .* log(1 - hypothesis));
    
    % Store values for plotting
    costHistory(iteration) = cost;
    classificationErrorHistory(iteration) = classificationError;
    
    % Check for convergence based on the change in theta
    if norm(theta - previousTheta) < tolerance
        break;
    end
    
    % Update previousTheta for the next iteration
    previousTheta = theta;
end

% Display the number of iterations needed for convergence
fprintf('Number of iterations for convergence: %d\n', iteration - 1);

% Plot classification error and empirical risk
figure;
subplot(2, 1, 1);
plot(1:iteration - 1, classificationErrorHistory(1:iteration - 1));
xlabel('Iterations');
ylabel('Classification Error');
title('Binary Classification Error');

subplot(2, 1, 2);
plot(1:iteration - 1, costHistory(1:iteration - 1));
xlabel('Iterations');
ylabel('Empirical Risk');
title('Logistic Loss (Empirical Risk)');

% Plot the resulting linear decision boundary
figure;
X_out = X(Y == 0, :, :);
disp(X(1, 1))

X_out1 = X(Y == 1, :, :);
XX = (-theta(3)-X(:, 1) * theta(1)) / theta(2);
scatter(X_out(:, 1),X_out(:, 2),'g')
hold on

scatter(X_out1(:, 1),X_out1(:, 2),'r')
hold on

plot(X(:, 1),XX,'k')
title("Decision Boundary");
legend('0','1',"Linear Decision Boundary")

