#ifndef PROCESS_H
#define PROCESS_H

#include <random>
#include <vector>

class NeuralNetwork {
public:
  NeuralNetwork(const std::vector<int> &layers);
  std::vector<double> predict(const std::vector<double> &inputs);
  std::vector<double>
  train_autoencoder(const std::vector<std::vector<double>> &data, int epochs,
                    double learning_rate);
  std::vector<double>
  train_classifier(const std::vector<std::vector<double>> &data,
                   const std::vector<std::vector<double>> &labels, int epochs,
                   double learning_rate);

private:
  std::vector<int> layer_sizes;
  std::vector<std::vector<std::vector<double>>> weights;
  std::vector<std::vector<double>> biases;

  // Momentum states
  std::vector<std::vector<std::vector<double>>> prev_weight_updates;
  std::vector<std::vector<double>> prev_bias_updates;

  std::vector<std::vector<double>> activations;
  std::vector<std::vector<double>> sums;
  std::mt19937 gen;
  double learning_rate;
  double momentum = 0.9;

  void initialize_weights();
  std::vector<double> feedforward(const std::vector<double> &inputs);
  void update_weights(const std::vector<std::vector<double>> &deltas);

  static double sigmoid(double x);
  static double sigmoid_derivative(double x);
  static std::vector<double> softmax(const std::vector<double> &z);

  void compute_z_score_params(const std::vector<std::vector<double>> &data,
                              std::vector<double> &mean,
                              std::vector<double> &std_dev);
  std::vector<double> normalize(const std::vector<double> &input,
                                const std::vector<double> &mean,
                                const std::vector<double> &std_dev);
};

#endif 
