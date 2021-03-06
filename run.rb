#!/usr/bin/env ruby

require 'optparse'
# require 'pry'
require './Preprocess.rb'
require './ANN.rb'

# default parameters
file_path = './Datasets/iris/iris.data.txt' # mandatory
perceptron_units = [] # optional
num_of_fold = 10 # optional
epoch_max = 2000 # optional
learning_rate = 0.1 # optional
momentum_rate = 0.05 # optional

# parser
OptionParser.new do |parser|
  parser.on("-i file_path", "--INPUT file_path", String, "INPUT CSV FILE CONTAINING LEARNING SAMPLES") do |f_path|
    file_path = f_path
  end
  parser.on("-u units", "--UNITS units", Array, "UNITS OF EACH LAYER") do |p_units|
    p_units.each_with_index {|val, index| p_units[index] = p_units[index].to_i}
    perceptron_units = p_units
  end
  # parser.on("-fold num_of_fold", "[OPTIONAL] NUMBER OF FOLDS IN CROSS VALIDATION") do |n_fold|
  #   num_of_fold = n_fold.to_i
  # end
  parser.on("-epoch num_of_epoch", "[OPTIONAL] NUMBER OF MAXIMUM EPOCH") do |iter_max|
    epoch_max = iter_max.to_i
  end
  parser.on("-l learning_rate", "[OPTIONAL] LEARNING RATE") do |l_rate|
    learning_rate = l_rate.to_f
  end
  parser.on("-m momentum_rate", "[OPTIONAL] MOMENTUM RATE, 0 IF NO MOMENTUM IS WANTED") do |m_rate|
    momentum_rate = m_rate.to_f
  end
end.parse!

# Data Preprocessing
pre = Preprocess.new(file_path)
data, var_type, mapping = pre.get_results()
perceptron_units = [data[0].size-1] + perceptron_units
perceptron_units << mapping.last.keys.size

# 10-fold-cross-validation
accuracies = []
iterations = []
(0..num_of_fold-1).each do |i|

  puts "====================================================================================="

  # divide the dataset into training_set test_set and validation_set (6:3:1)
  training_set, validation_set, test_set = Preprocess.divide_dataset(data, i, num_of_fold)

  puts "training set size = " + training_set.size.to_s
  puts "validation set size = " + validation_set.size.to_s
  puts "test set size = " + test_set.size.to_s

  # separate attributes and targets
  attr_train, target_train = Preprocess.separate_attr_and_target(training_set)
  attr_validation, target_validation = Preprocess.separate_attr_and_target(validation_set)
  attr_test, target_test = Preprocess.separate_attr_and_target(test_set)

  # training and evaluation
  weights, mapping, iteration = ann_learn(attr_train, target_train, attr_validation, target_validation, perceptron_units, epoch_max, learning_rate, momentum_rate)
  accuracies << ann_evaluate(attr_test, target_test, weights, mapping)
  iterations << iteration

  # reporting accuracy on each fold
  puts "--------------------------------------------------------------------------------------"
  puts "ACCURACY ON TEST SET = #{accuracies.last}"
  print "=====================================================================================\n\n\n"
end

puts "=====================================Result=========================================="
puts "Number of units of each layer = #{perceptron_units}"
puts "Number of iterations to stop = #{iterations}"
puts "LEARNING_RATE = #{learning_rate}, MOMENTUM_RATE = #{momentum_rate}"
puts "TEN-FOLD ACCURACIES = #{accuracies}"
puts "MEAN ACCURACIES = #{mean(accuracies)}"
puts "CONVIDENCE INTERVAL (0.95) = #{confidence_interval(accuracies)}"
