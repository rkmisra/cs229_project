function [theta] = main()
  [X_train,Y_train,X_test,Y_test] = load_planets_data('../data/habitable_planets_detailed_list.csv','../data/non_habitable_planets_confirmed_detailed_list.csv','../data/feature_index.csv');
  %[error] = logistic_regression(X_train,Y_train, X_test, Y_test);
  [error] = svm(X_train,Y_train, X_test, Y_test);

  fprintf('Test set error is %d  \n', error*100);
  fflush(stdout);


  [error] = svm(X_train,Y_train, X_train, Y_train);

  fprintf('Training error is %d  \n', error*100);
  fflush(stdout);

  [error] = svm(X_train,Y_train, [X_train; X_test], [Y_train; Y_test]);

  fprintf('Whole data error is %d  \n', error*100);
  fflush(stdout);
  
end

function[X_train, Y_train, X_test, Y_test] = load_planets_data(habitable_planets_training_file, non_habitable_planets_training_file, feature_selection_file)

  habitable = csvread(habitable_planets_training_file);
  non_habitable = csvread(non_habitable_planets_training_file);
  features_index = csvread(feature_selection_file);
  
  training_percentage = 0.7;
  
  habitable_train_size = floor(size(habitable, 1) * training_percentage);
  non_habitable_train_size = floor(size(non_habitable, 1) * training_percentage);
  
  %non_habitable_train_size =100;
  
  habitable_train = habitable(1:habitable_train_size,:);
  habitable_test = habitable(habitable_train_size+1:end,:);
  non_habitable_train = non_habitable (1:non_habitable_train_size,:);
  non_habitable_test = non_habitable (non_habitable_train_size+1:end,:);
  %non_habitable_test = non_habitable (non_habitable_train_size+1:non_habitable_train_size+200,:);
  
  % now select only relevant features
  habitable_train_features = ones(habitable_train_size,1);
  habitable_test_features = ones(size(habitable,1)-habitable_train_size,1);
  non_habitable_train_features = -ones(non_habitable_train_size,1);
  non_habitable_test_features = -ones(size(non_habitable,1)-non_habitable_train_size,1);
  %non_habitable_test_features = -ones(200,1);
  for index = features_index
    habitable_train_features = [habitable_train_features , habitable_train(:,index)];
    habitable_test_features = [habitable_test_features , habitable_test(:,index)];
    non_habitable_train_features = [non_habitable_train_features , non_habitable_train(:,index)];
    non_habitable_test_features = [non_habitable_test_features , non_habitable_test(:,index)];
  end
  
  %merge both training samples
  training_data = [habitable_train_features ; non_habitable_train_features];
  test_data = [habitable_test_features;non_habitable_test_features];
  
  %adding intercept too
  %X_train = [ones(size(training_data,1), 1),training_data(:,2:end)];
  X_train = [training_data(:,2:end)];
  Y_train = training_data(:,1);
  
  dlmwrite('train_data.txt', training_data, ' ');
  dlmwrite('test_data.txt',test_data, ' ');
  
  %X_test = [ones(size(test_data,1), 1),test_data(:,2:end)];
  X_test = [test_data(:,2:end)];
  Y_test = test_data(:,1);

  %libsvmwrite('train_data.txt', Y_train, X_train);
  %libsvmwrite('test_data.txt', Y_test , X_test);
  
end

function[error] = logistic_regression(X, Y, X_test, Y_test)
  theta = zeros(size(X, 2), 1);
  learning_rate = 0.01;
  for i = 1:10^7
    prev_theta = theta;
    grad = compute_gradient(X, Y, theta);
    theta = theta - learning_rate * grad;
    if mod(i, 10000) == 0
     fprintf('Iteration %d completed \n', i);
     fflush(stdout);
     %fprintf('Theta =  %f  \n', theta);
     %fflush(stdout);
    end
    if norm(theta - prev_theta) < 10^-4
     fprintf('converged in %d iterations \n', i-1);
     fflush(stdout);
     break;
    end
  end
  
  test_result = 1 ./(1 + exp(-theta' * X_test'));
  test_prediction = ones(size(test_result));
  error = 0;
  for i = 1:size(test_result,2)
    if test_result(i) < 0.5
      if Y_test(i) == 1
        error = error + 1;
      else
      end
    else
      if Y_test(i) == -1
        error = error + 1;
      else
      end
    end
  end  
end

function[grad] = compute_gradient(X, y, theta)
  grad = (-1 / size(X, 1)) * X' * (y ./(1 + exp((X * theta).* y)));
end


function[error] = svm(Xtrain, ytrain, XTest, ytest)
    rand ("seed", 123);
    m_train = size(Xtrain, 1);
%    ytrain = (2 * trainCategory - 1)';
%    Xtrain = 1.0 * (Xtrain > 0);

    squared_X_train = sum(Xtrain.^2, 2);
    gram_train = Xtrain * Xtrain';
    tau = 8;

    % Get full training matrix for kernels using vectorized code.
    Ktrain = full(exp(-(repmat(squared_X_train, 1, m_train) ...
                        + repmat(squared_X_train', m_train, 1) ...
                        - 2 * gram_train) / (2 * tau^2)));

    lambda = 1 / (64 * m_train);
    num_outer_loops = 40;
    alpha = zeros(m_train, 1);

    avg_alpha = zeros(m_train, 1);
    Imat = eye(m_train);

    count = 0;
    for ii = 1:(num_outer_loops * m_train)
      count = count + 1;
      ind = ceil(rand * m_train);
      margin = ytrain(ind) * Ktrain(ind, :) * alpha;
      g = -(margin < 1) * ytrain(ind) * Ktrain(:, ind) + ...
          m_train * lambda * (Ktrain(:, ind) * alpha(ind));
      % g(ind) = g(ind) + m_train * lambda * Ktrain(ind,:) * alpha;
      alpha = alpha - g / sqrt(count);
      avg_alpha = avg_alpha + alpha;
    end
    avg_alpha = avg_alpha / (num_outer_loops * m_train);
    error = svm_test(m_train, Xtrain, squared_X_train, avg_alpha, tau, XTest, ytest);
end

function[test_error] = svm_test(m_train, Xtrain, squared_X_train, avg_alpha, tau, Xtest, ytest)
  squared_X_test = sum(Xtest.^2, 2);
  m_test = size(Xtest, 1);
%  ytest = (2 * testCategory - 1)';
  gram_test = Xtest * Xtrain';
  Ktest = full(exp(-(repmat(squared_X_test, 1, m_train) ...
                     + repmat(squared_X_train', m_test, 1) ...
                     - 2 * gram_test) / (2 * tau^2)));

  % preds = Ktest * alpha;

  % fprintf(1, 'Test error rate for final alpha: %1.4f\n', ...
  %         sum(preds .* ytest <= 0) / length(ytest));

  preds = Ktest * avg_alpha;
  test_error = sum(preds .* ytest <= 0) / length(ytest);
end