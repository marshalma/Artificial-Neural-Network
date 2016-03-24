require 'matrix'

def ann_learn(examples, target, examples_val, target_val, units, epoch_max = 2000, learning_rate = 0.1, momentum_rate = 0.05)
  num_input_unit = units.first
  num_output_unit = units.last
  num_of_layers = units.size
  num_of_hidden_layers = num_of_layers - 2

  mapping = encode_target!(target, units.last)

  # initialize weight vectors randomly.
  weights = []
  last_delta_weights = [] # for the purpose of adding momentum to back prop algorithm
  (0..num_of_layers-2).each { |i| weights << ( Matrix.build(units[i]+1, units[i+1]) {rand} ).to_a }
  (0..num_of_layers-2).each { |i| last_delta_weights << ( Matrix.build(units[i]+1, units[i+1]) {0} ).to_a }

  # implementation of Back Propagation Algorithm
  # main loop
  max_accuracy = 0
  max_accuracy_iter = 0
  max_accuracy_weights = nil
  (0..epoch_max-1).each do |iter|
    # stop criteria
    break if 3 * max_accuracy_iter < iter && iter.to_f > 0.2 * epoch_max.to_f

    # puts weights.to_s
    examples.each_with_index do |example, index|
      # puts index.to_s
      activation = activation_calc(example, weights)
      deltas = []

      # output unit delta
      deltas << output_delta_calc(activation.last, target[index])

      # hidden unit delta
      (num_of_hidden_layers-1).downto(0) do |i|
        deltas << hidden_delta_cal(activation[i], weights[i+1], deltas.last)
      end

      # reverse deltas
      deltas.reverse!

      # update weights
      last_delta_weights = update_weights(weights, last_delta_weights, deltas, [example] + activation, learning_rate, momentum_rate)
    end

    # track TRAINING AND VALIDATION MSE
    raw_predictions, predictions = ann_predict(examples, weights, mapping)
    raw_predictions_val, predictions_val = ann_predict(examples_val, weights, mapping)
    accuracy_val = ann_evaluate(examples_val, target_val, weights, mapping)

    # maintain the weights of the epoch that has the maximum accuracy on validation set
    if accuracy_val > max_accuracy
      max_accuracy = accuracy_val
      max_accuracy_iter = iter
      max_accuracy_weights = weights
    end

    # print out result of this epoch
    puts "--------------------------------------------------------------------------------------"
    puts "epoch =  #{iter.to_s}, TRAINING MSE = #{mse_calc raw_predictions, target}, VALIDATION MSE = #{mse_calc raw_predictions_val, target_val}"
    puts "current accuracy = #{accuracy_val}, highest accuracy epoch = #{max_accuracy_iter}, highest accuracy = #{max_accuracy}"

  end
  [max_accuracy_weights, mapping]
end

def ann_evaluate(examples, target, weights, mapping)
  raw_prediction, predictions = ann_predict(examples, weights, mapping)

  num_of_accurate_predictions = 0
  predictions.each_with_index {|val,i| num_of_accurate_predictions += 1 if val == target[i]}
  return num_of_accurate_predictions.to_f / predictions.size.to_f
end

def ann_predict(examples, weights, mapping)
  raw = []

  examples.each do |input|
    activation = activation_calc input, weights
    raw << activation.last
  end

  predictions = Marshal.load(Marshal.dump(raw))
  predictions.each do |prediction|
    max = prediction[0]
    index = 0
    prediction.size.times do |i|
      if max < prediction[i]
        max = prediction[i]
        index = i
      end
    end
    prediction.size.times do |i|
      prediction[i] = 0 if index != i
      prediction[i] = 1 if index == i
    end
  end

  decode_target! predictions, mapping
  # binding.pry
  [raw, predictions]
end

def activation_calc(input, weights)
  result = []
  weights.each do |weight|
    input = sigmoid( (Matrix.row_vector([1] + input) * Matrix.rows(weight)).to_a.first )
    result << input
  end
  result
end

    def sigmoid(vector)
      result = []
      vector.each do |item|
        result << 1/(1 + Math.exp(-item))
      end
      result
    end

def output_delta_calc(o, t)
  result = []
  o.each_with_index {|output,k| result << output * (1 - output) * (t[k] - output)}
  result
end

def hidden_delta_cal(o, weight, delta)
  result = []
  o.each_with_index do |output, h|
    weighted_sigma = 0
    delta.each_with_index {|val,j| weighted_sigma += weight[h][j] * val}
    result << output * (1 - output) * weighted_sigma
  end
  result
end

def update_weights(weights, last_deltas_weights, deltas, activations, learning_rate, momentum_rate)
  # puts deltas.to_s
  # puts activations.to_s
  delta_weights = Marshal.load(Marshal.dump(last_deltas_weights))
  weights.each_with_index do |weight, iter|
    weight.each_with_index do |row, i|
      row.each_with_index do |val, j|
        if i == 0
          delta_weights[iter][i][j] = learning_rate * deltas[iter][j] + momentum_rate * last_deltas_weights[iter][i][j]
        else
          delta_weights[iter][i][j] = learning_rate * deltas[iter][j] * activations[iter][i-1] + momentum_rate * last_deltas_weights[iter][i][j]
        end
        weights[iter][i][j] += delta_weights[iter][i][j]
      end
    end
  end
  delta_weights
end

def encode_target!(target, num_of_output_units)
  mapping = Hash.new
  target.each_with_index do |val, index|
    subresult = Array.new(num_of_output_units, 0)
    subresult[val] = 1
    mapping[val] = subresult if !mapping.has_key? val
    target[index] = subresult;
  end
  mapping
end

def decode_target!(target, mapping)
  target.each_with_index do |val, index|
    target[index] = mapping.key(val)
  end
end

def mse_calc(predictions, targets)
  result = 0
  predictions.size.times do |i|
    subresult = 0
    predictions[i].size.times {|j| subresult += (predictions[i][j] - targets[i][j]) ** 2}
    result += 0.5 * subresult
  end
  result /= predictions.size.to_f
end

# SIMPLE TESTS (be advised that some of the methods are obsolete)
# puts encode_target([3],5).to_s # test of encode_target
# puts (decode_target!([[0,0,0,1,0,0]], encode_target!([3,4,5],6)) ).to_s
# examples = [[5.7, 3.8, 1.7, 0.3, 0], [6.5, 3.0, 5.2, 2.0, 2], [6.9, 3.2, 5.7, 2.3, 2], [5.1, 3.4, 1.5, 0.2, 0], [6.3, 2.8, 5.1, 1.5, 2], [7.7, 2.6, 6.9, 2.3, 2], [5.7, 3.0, 4.2, 1.2, 1], [5.4, 3.4, 1.7, 0.2, 0], [7.3, 2.9, 6.3, 1.8, 2], [6.2, 2.9, 4.3, 1.3, 1], [5.8, 2.7, 5.1, 1.9, 2], [4.9, 2.5, 4.5, 1.7, 2], [5.1, 3.5, 1.4, 0.3, 0], [6.3, 2.5, 5.0, 1.9, 2], [4.3, 3.0, 1.1, 0.1, 0], [5.0, 3.0, 1.6, 0.2, 0], [6.7, 2.5, 5.8, 1.8, 2], [4.4, 3.0, 1.3, 0.2, 0], [5.0, 3.2, 1.2, 0.2, 0], [6.7, 3.0, 5.2, 2.3, 2], [4.4, 2.9, 1.4, 0.2, 0], [5.1, 3.8, 1.9, 0.4, 0], [5.4, 3.9, 1.3, 0.4, 0], [5.7, 2.6, 3.5, 1.0, 1], [7.2, 3.6, 6.1, 2.5, 2], [5.0, 2.0, 3.5, 1.0, 1], [5.5, 4.2, 1.4, 0.2, 0], [5.8, 2.8, 5.1, 2.4, 2], [6.7, 3.0, 5.0, 1.7, 1], [6.7, 3.3, 5.7, 2.5, 2], [7.9, 3.8, 6.4, 2.0, 2]]
# examples.each {|item| puts item.to_s}
# mapping = normalize!(examples)
# examples.each {|item| puts item.to_s}
# examples1 = [[5.7, 3.8, 1.7, 0.3, 0], [6.5, 3.0, 5.2, 2.0, 2], [6.9, 3.2, 5.7, 2.3, 2], [5.1, 3.4, 1.5, 0.2, 0], [6.3, 2.8, 5.1, 1.5, 2], [7.7, 2.6, 6.9, 2.3, 2], [5.7, 3.0, 4.2, 1.2, 1], [5.4, 3.4, 1.7, 0.2, 0], [7.3, 2.9, 6.3, 1.8, 2], [6.2, 2.9, 4.3, 1.3, 1], [5.8, 2.7, 5.1, 1.9, 2], [4.9, 2.5, 4.5, 1.7, 2], [5.1, 3.5, 1.4, 0.3, 0], [6.3, 2.5, 5.0, 1.9, 2], [4.3, 3.0, 1.1, 0.1, 0], [5.0, 3.0, 1.6, 0.2, 0], [6.7, 2.5, 5.8, 1.8, 2], [4.4, 3.0, 1.3, 0.2, 0], [5.0, 3.2, 1.2, 0.2, 0], [6.7, 3.0, 5.2, 2.3, 2], [4.4, 2.9, 1.4, 0.2, 0], [5.1, 3.8, 1.9, 0.4, 0], [5.4, 3.9, 1.3, 0.4, 0], [5.7, 2.6, 3.5, 1.0, 1], [7.2, 3.6, 6.1, 2.5, 2], [5.0, 2.0, 3.5, 1.0, 1], [5.5, 4.2, 1.4, 0.2, 0], [5.8, 2.8, 5.1, 2.4, 2], [6.7, 3.0, 5.0, 1.7, 1], [6.7, 3.3, 5.7, 2.5, 2], [7.9, 3.8, 6.4, 2.0, 2]]
# normalize_with_range! examples1, mapping[0], mapping[1]
# examples1.each {|item| puts item.to_s}
# puts examples.to_s
# puts sigmoid([0,1,-1,1000,-1000]).to_s
# ann_learn([[1,2],[1,3],[2,3],[2,2],[1,1],[3,3],[3,1],[3,2],[2,1],[1,0],[3,0]], nil, [2,3,1])
