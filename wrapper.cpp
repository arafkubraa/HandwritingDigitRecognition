#include "Process.h"
#include <Python.h>
#include <vector>

// Python list std::vector<double>
bool PyList_To_Vector(PyObject *py_list, std::vector<double> &vec) {
  if (!PyList_Check(py_list)) {
    return false;
  }
  vec.resize(PyList_Size(py_list));
  for (Py_ssize_t i = 0; i < PyList_Size(py_list); ++i) {
    PyObject *item = PyList_GetItem(py_list, i);
    vec[i] = PyFloat_AsDouble(item);
    if (PyErr_Occurred()) {
      return false;
    }
  }
  return true;
}

// Python list std::vector<std::vector<double>>
bool PyListOfLists_To_Vector2D(PyObject *py_lol,
                               std::vector<std::vector<double>> &vec2d) {
  if (!PyList_Check(py_lol)) {
    return false;
  }
  vec2d.resize(PyList_Size(py_lol));
  for (Py_ssize_t i = 0; i < PyList_Size(py_lol); ++i) {
    PyObject *inner_list = PyList_GetItem(py_lol, i);
    if (!PyList_To_Vector(inner_list, vec2d[i])) {
      return false;
    }
  }
  return true;
}

typedef struct {
  PyObject_HEAD NeuralNetwork *nn;
} PyNeuralNetwork;

static int PyNeuralNetwork_init(PyNeuralNetwork *self, PyObject *args,
                                PyObject *kwds) {
  PyObject *py_layers;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &py_layers)) {
    return -1;
  }

  std::vector<int> layers;
  for (Py_ssize_t i = 0; i < PyList_Size(py_layers); ++i) {
    PyObject *item = PyList_GetItem(py_layers, i);
    layers.push_back(PyLong_AsLong(item));
  }

  self->nn = new NeuralNetwork(layers);
  return 0;
}

static void PyNeuralNetwork_dealloc(PyNeuralNetwork *self) {
  delete self->nn;
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyNeuralNetwork_train_autoencoder(PyNeuralNetwork *self,
                                                   PyObject *args) {
  PyObject *py_training_data;
  int epochs;
  double learning_rate;

  if (!PyArg_ParseTuple(args, "O!id", &PyList_Type, &py_training_data, &epochs,
                        &learning_rate)) {
    return NULL;
  }

  std::vector<std::vector<double>> training_data;
  if (!PyListOfLists_To_Vector2D(py_training_data, training_data)) {
    PyErr_SetString(PyExc_TypeError,
                    "Arguments must be lists of lists of numbers.");
    return NULL;
  }

  std::vector<double> history =
      self->nn->train_autoencoder(training_data, epochs, learning_rate);

  PyObject *py_history = PyList_New(history.size());
  for (size_t i = 0; i < history.size(); ++i) {
    PyList_SetItem(py_history, i, PyFloat_FromDouble(history[i]));
  }

  return py_history;
}

static PyObject *PyNeuralNetwork_train_classifier(PyNeuralNetwork *self,
                                                  PyObject *args) {
  PyObject *py_training_data, *py_training_labels;
  int epochs;
  double learning_rate;

  if (!PyArg_ParseTuple(args, "O!O!id", &PyList_Type, &py_training_data,
                        &PyList_Type, &py_training_labels, &epochs,
                        &learning_rate)) {
    return NULL;
  }

  std::vector<std::vector<double>> training_data, training_labels;
  if (!PyListOfLists_To_Vector2D(py_training_data, training_data) ||
      !PyListOfLists_To_Vector2D(py_training_labels, training_labels)) {
    PyErr_SetString(PyExc_TypeError,
                    "Arguments must be lists of lists of numbers.");
    return NULL;
  }

  std::vector<double> history = self->nn->train_classifier(
      training_data, training_labels, epochs, learning_rate);

  PyObject *py_history = PyList_New(history.size());
  for (size_t i = 0; i < history.size(); ++i) {
    PyList_SetItem(py_history, i, PyFloat_FromDouble(history[i]));
  }

  return py_history;
}

static PyObject *PyNeuralNetwork_predict(PyNeuralNetwork *self,
                                         PyObject *args) {
  PyObject *py_input_data;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &py_input_data)) {
    return NULL;
  }

  std::vector<double> input_data;
  if (!PyList_To_Vector(py_input_data, input_data)) {
    PyErr_SetString(PyExc_TypeError, "Argument must be a list of numbers.");
    return NULL;
  }

  std::vector<double> prediction = self->nn->predict(input_data);

  PyObject *py_prediction = PyList_New(prediction.size());
  for (size_t i = 0; i < prediction.size(); ++i) {
    PyList_SetItem(py_prediction, i, PyFloat_FromDouble(prediction[i]));
  }

  return py_prediction;
}

static PyMethodDef PyNeuralNetwork_methods[] = {
    {"train_autoencoder", (PyCFunction)PyNeuralNetwork_train_autoencoder,
     METH_VARARGS, "Train the Autoencoder"},
    {"train_classifier", (PyCFunction)PyNeuralNetwork_train_classifier,
     METH_VARARGS, "Train the Classifier"},
    {"predict", (PyCFunction)PyNeuralNetwork_predict, METH_VARARGS,
     "Predict output for an input"},
    {NULL} 
};

static PyTypeObject PyNeuralNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "my_process_module.NeuralNetwork",
    .tp_doc = "Neural Network object",
    .tp_basicsize = sizeof(PyNeuralNetwork),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyNeuralNetwork_init,
    .tp_dealloc = (destructor)PyNeuralNetwork_dealloc,
    .tp_methods = PyNeuralNetwork_methods,
};

static struct PyModuleDef my_process_module = {
    PyModuleDef_HEAD_INIT,
    "my_process_module",
    "A Python module for a C++ neural network.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit_my_process_module(void) {
  PyObject *m;
  if (PyType_Ready(&PyNeuralNetworkType) < 0)
    return NULL;

  m = PyModule_Create(&my_process_module);
  if (m == NULL)
    return NULL;

  Py_INCREF(&PyNeuralNetworkType);
  if (PyModule_AddObject(m, "NeuralNetwork", (PyObject *)&PyNeuralNetworkType) <
      0) {
    Py_DECREF(&PyNeuralNetworkType);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
