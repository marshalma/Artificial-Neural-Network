require 'csv'

class Preprocess
  def initialize(file_path)
    @data = CSV.read(file_path)
    @number_of_features = @data[0].size
    @number_of_examples = @data.size

    find_discrete_continuous_variables()
    convert_continuous_var()
    create_discrete_mapping()
    encode_discrete_var()
    shuffle()
    normalize!(@data, [@number_of_features-1])
  end

  def find_discrete_continuous_variables
    # :d_var, :c_var represents discrete and continuous variable
    @var_type = [];
    @data[0].each do |var|
      @var_type << (nan?(var) ? :d_var : :c_var)
    end
  end

  def convert_continuous_var
    @var_type.each_with_index do |type,index|
      next if type == :d_var
      @data.each do |item|
        item[index] = item[index].to_f
      end
    end
  end

  def create_discrete_mapping
    @mapping = []
    @var_type.each_with_index do |item,i|
      if item == :c_var
        @mapping << nil
      else
        all_values = find_all_values(i)
        this_mapping = Hash.new
        all_values.each_with_index do |val, index|
          this_mapping[val] = index
        end
        @mapping << this_mapping
      end
    end
  end

  def encode_discrete_var
    @var_type.each_with_index do |item,i|
      next if item != :d_var
      @data.size.times do |index|
        @data[index][i] = @mapping[i][@data[index][i]]
      end
    end
  end

  def shuffle
    @data.shuffle!
  end

  def get_results
    return [@data, @var_type, @mapping]
  end

  # HELPERS
  # check if a variable is not a number
  def nan?(str)
    str !~ /^\s*[+-]?((\d+_?)*\d+(\.(\d+_?)*\d+)?|\.(\d+_?)*\d+)(\s*|([eE][+-]?(\d+_?)*\d+)\s*)$/
  end

  def find_all_values(attr_index)
    all_values = Hash.new
    @data.each do |item|
      all_values[item[attr_index]] = true if !all_values.has_key? item
    end
    return all_values.keys
  end

  def normalize!(examples, without_index)
    num_of_attr = examples[0].size
    max_arr = []; min_arr = []
    (0..num_of_attr-1).each do |i|
      next if without_index.include? i

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

  def self.separate_attr_and_target(data)
    attribute = []
    target = []
    (0..data.size-1).each do |i|
      attribute << data[i][0..data[i].size-2]
      target << data[i].last
    end
    [attribute, target]
  end

  def self.divide_dataset(data, i, num_of_fold)
    num_in_one_fold = data.size / 10

    training_set = []
    validation_set = []
    test_set = []

    range = (i*num_in_one_fold..(i+1)*num_in_one_fold-1)
    data.size.times do |j|
      if range.include? j
        test_set << data[j]
      else
        training_set << data[j]
      end
    end

    split = training_set.size * 2 / 3
    validation_set = training_set[split..training_set.size-1]
    training_set = training_set[0..split-1]

    [training_set, validation_set, test_set]
  end

end
