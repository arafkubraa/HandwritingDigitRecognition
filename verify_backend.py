import my_process_module
import sys

def verify():
    print("Initializing NeuralNetwork...")
    nn = my_process_module.NeuralNetwork([2, 2, 2])
    
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    
    print("Training Phase 1: Autoencoder...")
    ae_hist = nn.train_autoencoder(X, 5, 0.1)
    print(f"AE Loss: {ae_hist}")

    print("Training Phase 2: Classifier...")
    clf_hist = nn.train_classifier(X, Y, 5, 0.1)
    
    print(f"Returned types: {type(ae_hist)}, {type(clf_hist)}")
    print(f"Returned values: {clf_hist}")
    
    if isinstance(clf_hist, list) and len(clf_hist) == 5:
        print("SUCCESS: History returned correctly.")
    else:
        print("FAILURE: Incorrect return type or structure.")
        sys.exit(1)

if __name__ == "__main__":
    verify()
