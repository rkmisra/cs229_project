function [selected_features_set] = forward_feature_search()
  selected_features_set = []();
  error_rate = []();
  
  %these are indexes related with planatory or steller data
  included_indexes = [16,31,40,43,...
                      46,50,53,56,59,62,65,77,...
                      84,85,88,91,94,97,100,103];

  habitable = csvread('../data/habitable_planets_detailed_list.csv');
  non_habitable = csvread('../data/non_habitable_planets_confirmed_detailed_list.csv');

  last_error = intmax('uint32');
  
  for k = 1:length(included_indexes)
    i = included_indexes(k);
    % add one remaining feature at time and find min error from test.
    selected_features_set_size = length(selected_features_set);
    min_error = intmax('uint32');
    best_feature_index = 0;
    for l = 1:length(included_indexes)
      j = included_indexes(l);
      if ismember(j , selected_features_set) == false
        selected_features_set(selected_features_set_size+1) = j;
        printf("Testing features %d \n",selected_features_set);
        fflush(stdout);
        [X_train, Y_train, X_dev, Y_dev, X_test,Y_test] = load_planets_data(habitable,non_habitable,selected_features_set);
        
        [error_dev] = svm(X_train, Y_train, X_dev, Y_dev);
          
        printf('Dev set error is %d %%\n', error_dev*100);
        fflush(stdout);  
          
        if error_dev < min_error
          min_error = error_dev;
          best_feature_index = j;
        end
      end
    end
    
    if last_error <= min_error 
      printf("Error improvement dropped to 0 \n");
      selected_features_set = selected_features_set(1:length(selected_features_set)-1);
      break;
    end
    
    selected_features_set(selected_features_set_size+1) = best_feature_index;
    error_rate(length(error_rate) + 1) = min_error; 
    last_error = min_error;
  end
  printf('Selected feature indexes is %d \n', selected_features_set);
  printf('Error rate drop is %d \n', error_rate);
  fflush(stdout); 
 
  [X_train, Y_train, X_dev, Y_dev, X_test, Y_test] = load_planets_data(habitable, non_habitable, selected_features_set);
  [error] = svm(X_train,Y_train, X_test, Y_test);
  
  fprintf('Test set error is %d  \n', error*100);
  fflush(stdout);
 
end

function[X_train, Y_train, X_dev, Y_dev, X_test, Y_test] = load_planets_data(habitable, non_habitable, selected_features_set)

  training_percentage = 0.5;
  dev_percentage = 0.2;
  
  habitable_train_size = floor(size(habitable, 1) * training_percentage);
  habitable_dev_size = floor(size(habitable, 1) * dev_percentage);
  non_habitable_train_size = floor(size(non_habitable, 1) * training_percentage);
  non_habitable_dev_size = floor(size(non_habitable, 1) * dev_percentage);
  
  habitable_train = habitable(1:habitable_train_size,:);
  habitable_dev = habitable(habitable_train_size+1:habitable_train_size+habitable_dev_size,:);
  habitable_test = habitable(habitable_train_size+habitable_dev_size+1:end,:);
  
  non_habitable_train = non_habitable (1:non_habitable_train_size,:);
  non_habitable_dev = non_habitable (non_habitable_train_size+1:non_habitable_train_size+non_habitable_dev_size,:);
  non_habitable_test = non_habitable (non_habitable_train_size+non_habitable_dev_size+1:end,:);
  
  habitable_train_features = ones(habitable_train_size,1);
  habitable_dev_features = ones(habitable_dev_size,1);
  habitable_test_features = ones(size(habitable,1)-(habitable_train_size + habitable_dev_size),1);
  
  non_habitable_train_features = -ones(non_habitable_train_size,1);
  non_habitable_dev_features = -ones(non_habitable_dev_size,1);
  non_habitable_test_features = -ones(size(non_habitable,1)-(non_habitable_train_size+non_habitable_dev_size),1);
  
  for i = 1:length(selected_features_set)
    index = selected_features_set(i);
    habitable_train_features = [habitable_train_features , habitable_train(:,index)];
    habitable_dev_features = [habitable_dev_features , habitable_dev(:,index)];
    habitable_test_features = [habitable_test_features , habitable_test(:,index)];
  
    non_habitable_train_features = [non_habitable_train_features , non_habitable_train(:,index)];
    non_habitable_dev_features = [non_habitable_dev_features , non_habitable_dev(:,index)];
    non_habitable_test_features = [non_habitable_test_features , non_habitable_test(:,index)];
  end
  
  %merge both training samples
  training_data = [habitable_train_features ; non_habitable_train_features];

  dev_data = [habitable_train_features ; habitable_dev_features ; non_habitable_dev_features];
  
  test_data = [habitable_test_features;non_habitable_test_features];
  
  X_train = [training_data(:,2:end)];
  Y_train = training_data(:,1);
  
  X_dev = [dev_data(:,2:end)];
  Y_dev = dev_data(:,1);
  
  X_test = [test_data(:,2:end)];
  Y_test = test_data(:,1);
  
end

function[error] = svm(Xtrain, ytrain, XTest, ytest)
    rand ("seed", 123);
    m_train = size(Xtrain, 1);
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
      alpha = alpha - g / sqrt(count);
      avg_alpha = avg_alpha + alpha;
    end
    avg_alpha = avg_alpha / (num_outer_loops * m_train);
    error = svm_test(m_train, Xtrain, squared_X_train, avg_alpha, tau, XTest, ytest);
end

function[test_error] = svm_test(m_train, Xtrain, squared_X_train, avg_alpha, tau, Xtest, ytest)
  squared_X_test = sum(Xtest.^2, 2);
  m_test = size(Xtest, 1);
  gram_test = Xtest * Xtrain';
  Ktest = full(exp(-(repmat(squared_X_test, 1, m_train) ...
                     + repmat(squared_X_train', m_test, 1) ...
                     - 2 * gram_test) / (2 * tau^2)));

  preds = Ktest * avg_alpha;
  
  % lets calculate error for -1 and + 1 labels seperately
  plus_1_vec = ones(length(find(ytest == 1)),1);
  minus_1_vec = -ones(length(find(ytest == -1)),1);
  test_error_plus_1 = sum(preds (1:length(plus_1_vec) ).* plus_1_vec <= 0) / length(plus_1_vec);
  test_error_minus_1 = sum(preds (length(plus_1_vec)+1:end) .* minus_1_vec <= 0) / length(minus_1_vec);
  
  test_error = (test_error_plus_1 + test_error_minus_1)/2;
end