function [theta] = main()
  [X_train,Y_train,X_test,Y_test] = load_planets_data('../data/habitable_planets_detailed_list.csv','../data/non_habitable_planets_confirmed_detailed_list.csv','../data/feature_index.csv');
  [theta] = logistic_regression(X_train,Y_train);

  fprintf('%f \n', theta);
  fflush(stdout);
  
  [error, Y_predicted] = test_logistic_regression(X_test, Y_test, theta);
  
  fprintf('Test set error is %d  \n', error*100);
  fflush(stdout);

  [error] = test_logistic_regression(X_train, Y_train, theta);

  fprintf('Training error is %d  \n', error*100);
  fflush(stdout);

  [error] = test_logistic_regression([X_train; X_test], [Y_train; Y_test], theta);

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

function[theta] = logistic_regression(X, Y, X_test, Y_test)
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
end

function [error, test_prediction] = test_logistic_regression(X_test, Y_test, theta)
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
  error = error/size(test_result,2) 
end

function[grad] = compute_gradient(X, y, theta)
  grad = (-1 / size(X, 1)) * X' * (y ./(1 + exp((X * theta).* y)));
end