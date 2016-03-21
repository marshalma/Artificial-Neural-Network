require 'matrix'

def ann_learn(examples, target, units, epoch = 10000, learning_rate=0.1)
  num_input_unit = units.first
  num_output_unit = units.last
  num_of_layers = units.size
  num_of_hidden_layers = num_of_layers - 2

  max_arr, min_arr = normalize!(examples)
  mapping = encode_target!(target)

  # initialize weight vectors randomly.
  weights = []
  (0..num_of_layers-2).each { |i| weights << ( Matrix.build(units[i]+1, units[i+1]) {rand} ).to_a }

  # implementation of Back Propagation Algorithm
  # main loop
  (0..epoch-1).each do |iter|
    examples.each_with_index do |example, index|
      activation = calc_activation(example, weights)
      deltas = []

      # output unit delta
      deltas << output_delta_calc(activation.last, target[index])

      # hidden unit delta
      (num_of_hidden_layers-1).downto(0) do |i|
        deltas << hidden_delta_cal(activation[i], weight[i+1], deltas.last)
      end

      # reverse deltas
      deltas.reverse!

      # update weights
      update_weights(weights, deltas, [examples] + activations, learning_rate)
     end
  end
  [weights, max_arr, min_arr, mapping]
end

def ann_evaluate(examples, target, weights, max_arr, min_arr, mapping)
  predictions = ann_predict(examples, weights, max_arr, min_arr, mapping)
  num_of_accurate_predictions = 0
  predictions.each_with_index {|val,i| num_of_accurate_predictions += 1 if val == target[i]}
  return num_of_accurate_predictions.to_f / predictions.size.to_f
end

def ann_predict(examples, weights, max_arr, min_arr, mapping)
  normalize_with_range! examples, max_arr, min_arr
  predictions = []
  examples.each do |input|
    activation = activation_calc input weights
    predictions << activation.last
  end
  decode_target! predictions, mapping
  predictions
end

def activation_calc(input, weights)
  result = []
  weights.each do |weight|
    input = sigmoid( (Matrix.row_vector([1] + input) * Matrix.rows(weight)).to_a )
    result << input
  end
  result
end

  def sigmoid(vector)
    result = []
    vector.each {|item| result << 1/(1 + Math.exp(-item))}
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

def update_weights(weights, deltas, activations, rate)
  weights.each_with_index do |weight, iter|
    weight.each_with_index do |row, j|
      row.each_with_index do |val, i|
        weights[iter][i][j] = rate * deltas[iter][j] * activations[iter]
      end
    end
  end
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

def normalize!(examples)
  num_of_attr = examples[0].size
  max_arr = []; min_arr = []
  (0..num_of_attr-1).each do |i|
    min = examples[0][i]
    max = examples[0][i]
    examples.each do |item|
      min = item[i] if item[i] < min
      max = item[i] if item[i] > max
    end
    max_arr << max.to_f; min_arr << min.to_f
    examples.each_with_index do |item,index|
      if min == max
        examples[index][i] = 0.to_f; next
      end
      examples[index][i] = (examples[index][i].to_f - min.to_f) / (max.to_f - min.to_f)
    end
  end
  [max_arr, min_arr]
end

def normalize_with_range!(examples, max_arr, min_arr)
  num_of_attr = examples[0].size
  (0..num_of_attr-1).each do |i|
    examples.each_with_index {|v,index| examples[index][i] = (examples[index][i].to_f - min_arr[i].to_f) / (max_arr[i].to_f - min_arr[i].to_f)}
  end
end


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
