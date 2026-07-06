#include "Process.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

NeuralNetwork::NeuralNetwork(const std::vector<int> &layers)
    : layer_sizes(layers) {
  std::random_device rd;
  gen = std::mt19937(rd());
  initialize_weights();

  
  activations.resize(layer_sizes.size());
  sums.resize(layer_sizes.size() - 1);
  for (size_t i = 0; i < layer_sizes.size(); ++i) {
    activations[i].resize(layer_sizes[i]);
    if (i < layer_sizes.size() - 1) {
      sums[i].resize(layer_sizes[i + 1]);
    }
  }
}

void NeuralNetwork::initialize_weights() {
  weights.resize(layer_sizes.size() - 1);
  biases.resize(layer_sizes.size() - 1);
  prev_weight_updates.resize(layer_sizes.size() - 1);
  prev_bias_updates.resize(layer_sizes.size() - 1);

  for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / layer_sizes[i]));
    weights[i].resize(layer_sizes[i + 1]);
    biases[i].resize(layer_sizes[i + 1]);
    prev_weight_updates[i].resize(layer_sizes[i + 1]);
    prev_bias_updates[i].resize(layer_sizes[i + 1]);

    for (int j = 0; j < layer_sizes[i + 1]; ++j) {
      weights[i][j].resize(layer_sizes[i]);
      prev_weight_updates[i][j].resize(layer_sizes[i], 0.0);
      for (int k = 0; k < layer_sizes[i]; ++k) {
        weights[i][j][k] = d(gen);
      }
      biases[i][j] = 0.0;
      prev_bias_updates[i][j] = 0.0;
    }
  }
}

double NeuralNetwork::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double NeuralNetwork::sigmoid_derivative(double x) { return x * (1.0 - x); }

std::vector<double> NeuralNetwork::softmax(const std::vector<double> &z) {
  std::vector<double> result;
  result.reserve(z.size());
  double max_z = *std::max_element(z.begin(), z.end());
  double sum = 0.0;
  for (double val : z)
    sum += std::exp(val - max_z);
  for (double val : z)
    result.push_back(std::exp(val - max_z) / sum);
  return result;
}

std::vector<double>
NeuralNetwork::feedforward(const std::vector<double> &inputs) {
  activations[0] = inputs;
  for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
    for (int j = 0; j < layer_sizes[i + 1]; ++j) {
      double z = 0.0;
      for (int k = 0; k < layer_sizes[i]; ++k)
        z += weights[i][j][k] * activations[i][k];
      z += biases[i][j];
      sums[i][j] = z;

      if (i == layer_sizes.size() - 2) { 
        activations[i + 1][j] =
            z; 
      } else {
        activations[i + 1][j] = sigmoid(z);
      }
    }
  }
  activations.back() = softmax(activations.back());
  return activations.back();
}

void NeuralNetwork::update_weights(
    const std::vector<std::vector<double>> &deltas) {
  for (int i = 0; i < layer_sizes.size() - 1; ++i) {
    for (int j = 0; j < layer_sizes[i + 1]; ++j) {
      for (int k = 0; k < layer_sizes[i]; ++k) {
        double grad = deltas[i][j] * activations[i][k];
        double update =
            (learning_rate * grad) + (momentum * prev_weight_updates[i][j][k]);
        weights[i][j][k] -= update; 
        prev_weight_updates[i][j][k] = update;
      }
      double grad_b = deltas[i][j];
      double update_b =
          (learning_rate * grad_b) + (momentum * prev_bias_updates[i][j]);
      biases[i][j] -= update_b;
      prev_bias_updates[i][j] = update_b;
    }
  }
}

void NeuralNetwork::compute_z_score_params(
    const std::vector<std::vector<double>> &data, std::vector<double> &mean,
    std::vector<double> &std_dev) {
  if (data.empty())
    return;
  int dim = data[0].size();
  mean.assign(dim, 0.0);
  std_dev.assign(dim, 0.0);

  for (const auto &sample : data) {
    for (int d = 0; d < dim; ++d)
      mean[d] += sample[d];
  }
  for (int d = 0; d < dim; ++d)
    mean[d] /= data.size();

  for (const auto &sample : data) {
    for (int d = 0; d < dim; ++d)
      std_dev[d] += std::pow(sample[d] - mean[d], 2);
  }
  for (int d = 0; d < dim; ++d)
    std_dev[d] = std::sqrt(std_dev[d] / data.size());
}

std::vector<double>
NeuralNetwork::normalize(const std::vector<double> &input,
                         const std::vector<double> &mean,
                         const std::vector<double> &std_dev) {
  std::vector<double> norm(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    norm[i] = (std_dev[i] != 0.0) ? (input[i] - mean[i]) / std_dev[i] : 0.0;
  }
  return norm;
}

// Autoencoder Training Input ->Enc -> Dec -> Output]
std::vector<double>
NeuralNetwork::train_autoencoder(const std::vector<std::vector<double>> &data,
                                 int epochs, double lr) {
  this->learning_rate = lr;
  std::vector<double> history;
  int input_dim = layer_sizes[0];
  int hidden_dim = layer_sizes[1]; // Encoder output

  // Decoder Weights (Hidden -> Input)
  std::vector<std::vector<double>> decoder_weights(
      input_dim, std::vector<double>(hidden_dim));
  std::vector<double> decoder_biases(input_dim, 0.0);

  std::mt19937 gen(1234);
  std::normal_distribution<double> d(0.0, std::sqrt(2.0 / hidden_dim));
  for (int i = 0; i < input_dim; ++i) {
    for (int j = 0; j < hidden_dim; ++j)
      decoder_weights[i][j] = d(gen);
  }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;
    for (const auto &sample : data) {
      // Encode
      activations[0] = sample;
      for (int j = 0; j < hidden_dim; ++j) {
        double z = 0.0;
        for (int k = 0; k < input_dim; ++k)
          z += weights[0][j][k] * activations[0][k];
        z += biases[0][j];
        sums[0][j] = z;
        activations[1][j] = sigmoid(z);
      }

      //Decode
      std::vector<double> reconstruction(input_dim);
      for (int j = 0; j < input_dim; ++j) {
        double z = 0.0;
        for (int k = 0; k < hidden_dim; ++k)
          z += decoder_weights[j][k] * activations[1][k];
        z += decoder_biases[j];
        reconstruction[j] =
            sigmoid(z); 
      }

      
      std::vector<double> delta_output(input_dim);
      for (int j = 0; j < input_dim; ++j) {
        double err = reconstruction[j] - sample[j];
        total_loss += err * err;
        delta_output[j] = err * sigmoid_derivative(reconstruction[j]);
      }

      // Decoder güncelleme
      for (int j = 0; j < input_dim; ++j) {
        for (int k = 0; k < hidden_dim; ++k) {
          decoder_weights[j][k] -= lr * delta_output[j] * activations[1][k];
        }
        decoder_biases[j] -= lr * delta_output[j];
      }

      std::vector<double> delta_hidden(hidden_dim);
      for (int j = 0; j < hidden_dim; ++j) {
        double err = 0.0;
        for (int k = 0; k < input_dim; ++k)
          err += delta_output[k] * decoder_weights[k][j];
        delta_hidden[j] = err * sigmoid_derivative(activations[1][j]);
      }

      // Encoder güncelleme (weights[0])
      for (int j = 0; j < hidden_dim; ++j) {
        for (int k = 0; k < input_dim; ++k) {
          weights[0][j][k] -= lr * delta_hidden[j] * activations[0][k];
        }
        biases[0][j] -= lr * delta_hidden[j];
      }
    }
    history.push_back(total_loss / data.size());
    if (epoch % 1 == 0)
      std::cout << "AE Epoch " << epoch + 1 << " Loss: " << history.back()
                << std::endl;
  }
  return history;
}


std::vector<double>
NeuralNetwork::train_classifier(const std::vector<std::vector<double>> &data,
                                const std::vector<std::vector<double>> &labels,
                                int epochs, double lr) {
  this->learning_rate = lr;
  std::vector<double> history;


  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
      feedforward(data[i]);

      std::vector<std::vector<double>> deltas(layer_sizes.size() - 1);
      for (size_t l = 0; l < deltas.size(); ++l)
        deltas[l].resize(layer_sizes[l + 1]);

      // Output Delta (Cross Entropy + Softmax) -> (P - Y)
      for (int j = 0; j < layer_sizes.back(); ++j) {
        deltas.back()[j] = activations.back()[j] - labels[i][j];
      }

      // Hidden Deltas
      for (int l = layer_sizes.size() - 2; l > 0; --l) {
        for (int j = 0; j < layer_sizes[l]; ++j) {
          double error = 0.0;
          for (int k = 0; k < layer_sizes[l + 1]; ++k) {
            error += deltas[l][k] * weights[l][k][j];
          }
          deltas[l - 1][j] = error * sigmoid_derivative(activations[l][j]);
        }
      }

      update_weights(deltas);

      // Loss
      for (size_t j = 0; j < activations.back().size(); ++j) {
        if (labels[i][j] == 1)
          total_loss -= std::log(activations.back()[j] + 1e-9);
      }
    }
    history.push_back(total_loss / data.size());
    if (epoch % 1 == 0)
      std::cout << "CLF Epoch " << epoch + 1 << " Loss: " << history.back()
                << std::endl;
  }
  return history;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &inputs) {
  return feedforward(inputs);
}
