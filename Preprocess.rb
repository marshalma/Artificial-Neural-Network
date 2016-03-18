class Preprocess
  def initialize(file_path)
    @data = CSV.read(file_path)
    @number_of_features = @data[0].size
    @number_of_examples = @data.size

    find_discrete_continuous_variables()
    create_discrete_mapping()
    encode_discrete_var()
    shuffle()
  end

  def find_discrete_continuous_variables
    # :d_var, :c_var represents discrete and continuous variable

    @data_var_type = [];
    @data[0].each do |var|
      @data_var_type << (nan?(var) ? :d_var : :c_var)
    end
  end

  def create_discrete_mapping
    @mapping = []
    @data_var_type.each_with_index do |item,i|
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
    @data_var_type.each_with_index do |item,i|
      next if item != :d_var
      (0..@data.size).each do |index|
        @data[index][i] = @mapping[i][@data[index][i]]
      end
    end
  end

  def shuffle
    @data.shuffle!
  end

  def get_results
    return [@data, @data_var_type, @mapping]
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

end

require 'csv'
file_path = './Datasets/iris/iris.data.txt'
a = Preprocess.new(file_path)
puts a.get_results.to_s
