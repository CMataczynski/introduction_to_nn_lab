import os
import urllib.request
import gzip
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from perceptron_utils import Perceptron, SGD, SigmoidActivation, BinaryCrossEntropyLoss

def compute_accuracy(output, target, threshold=0.5):
    """
    Compute accuracy given the output and target labels.

    Args:
        output (np.ndarray): Model output. Shape (batch_size, 1).
        target (np.ndarray): Target labels. Shape (batch_size, ).

    Returns:
        float: Accuracy as a percentage.
    """
    #TODO: Implement this function. Remember of the shapes of output and target.
    pass

class ModelTrainer:
    def __init__(self, model, loss_object) -> None:
        """
        Initialize ModelTrainer.

        Args:
            model: The model to train.
            loss_object: The loss function to use (e.g., MSE, CrossEntropy).
        """
        self.model = model
        self.loss_object = loss_object

    def train_batch(self, data_batch, target_batch):
        """Train the model on a single batch of data and corresponding labels.

        Args:
            data_batch (np.ndarray): Batch of input data. Shape (batch_size, input_size).
            target_batch (np.ndarray): Batch of corresponding labels. Shape (batch_size, ).

        Returns:
            tuple:
                float: The computed loss.
                float: The computed accuracy.
        """
        #TODO: Implement this function. Remember of the shapes of data_batch and target_batch.
        pass


    def train_model(self, train_data, train_target, test_data, test_target, epochs, batch_size):
        """
        Train the model on a dataset.

        Args:
            train_data (np.ndarray): Training input data.
            train_target (np.ndarray): Training target data.
            test_data (np.ndarray): Test input data.
            test_target (np.ndarray): Test target data.
            epochs (int): Number of epochs to train for.
            batch_size (int): Size of each batch.

        Returns:
            class Perceptron: Trained model.
        """
        plt.ion()
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        ax_loss, ax_accuracy = axs
        self.epoch_counter = []  # Store epoch numbers

        self.train_loss_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.train_accuracy_history = []

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            train_data, train_target = shuffle_data(train_data, train_target)
            train_accuracies = []
            train_losses = []
            for i in range(0, train_data.shape[0], batch_size):
                data_batch = train_data[i:i + batch_size]
                target_batch = train_target[i:i + batch_size]
                loss, acc = self.train_batch(data_batch, target_batch)
                train_losses.append(loss)
                train_accuracies.append(acc)
            
            train_loss = np.mean(train_losses)
            test_output = self.model.forward(test_data)
            test_loss, _ = self.loss_object.forward(test_output, test_target)
            test_accuracy = compute_accuracy(test_output, test_target)
            train_accuracy = np.mean(train_accuracies)
        
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            self.test_accuracy_history.append(test_accuracy)
            self.train_accuracy_history.append(train_accuracy)
            
            self.epoch_counter.append(epoch)  # Add epoch number
            update_progress_bar(pbar, epoch + 1, train_loss, test_loss, test_accuracy)
            
            #update training visualization
            self.update_training_visualization(ax_loss, ax_accuracy)

            fig.canvas.draw()
            plt.pause(0.01)
    
        # Turn off interactive mode after training
        plt.ioff()
        plt.show()
        return self.model

    def update_training_visualization(self, ax_loss, ax_accuracy):
        """
        Visualize the training process.
        """
        ax_loss.clear()
        ax_loss.plot(self.epoch_counter, self.train_loss_history, label='Train Loss')
        ax_loss.plot(self.epoch_counter, self.test_loss_history, label='Test Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()

        
        ax_accuracy.clear()
        ax_accuracy.plot(self.epoch_counter, self.train_accuracy_history, label='Train Accuracy')
        ax_accuracy.plot(self.epoch_counter, self.test_accuracy_history, label='Test Accuracy')
        ax_accuracy.set_xlabel('Epoch')
        ax_accuracy.set_ylabel('Accuracy (%)')
        ax_accuracy.legend()


def update_progress_bar(pbar, epoch, train_loss, test_loss, test_accuracy):
    """
    Update the progress bar with training information.

    Args:
        pbar: tqdm progress bar object.
        epoch (int): Current epoch.
        train_loss (float): Training loss.
        test_loss (float): Test loss.
        test_accuracy (float): Test accuracy.
    """
    pbar.set_description("Epoch: {}, Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(
        epoch, train_loss, test_loss, test_accuracy))




def shuffle_data(data, labels):
    """
    Shuffle the data and corresponding labels.

    Args:
        data (np.ndarray): Input data.
        labels (np.ndarray): Corresponding labels.

    Returns:
        np.ndarray: Shuffled data.
        np.ndarray: Shuffled labels.
    """
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]

def download_mnist(path):
    """
    Downloads the MNIST dataset files if they don't already exist in the specified path.

    Parameters:
    path (str): The directory where the dataset files will be stored.

    Returns:
    None
    """
    if not os.path.exists(path):
        os.mkdir(path)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for file in files:
        #check if file exists
        if os.path.exists(os.path.join(path, file)):
            print(f"File {file} already exists. Skipping download.")
            continue
        url = (base_url + file).format(**locals())
        print("Downloading " + url)
        try:
            urllib.request.urlretrieve(url, os.path.join(path, file))
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    print("Download complete.")

def load_mnist_binary(path, kind="train", targets=[0, 1]):
    """
    Loads the MNIST dataset from the specified path and returns images and labels. 
    Only images with labels in targets are returned.

    Parameters:
    path (str): The directory where the dataset files are stored.
    kind (str): "train" for training data, "test" for test data.

    Returns:
    images (numpy.ndarray): An array containing image data.
    labels (numpy.ndarray): An array containing corresponding labels.
    """
    if kind == "test":
        kind = "t10k"
    labels_path = os.path.join(path, "{}-labels-idx1-ubyte.gz".format(kind))
    images_path = os.path.join(path, "{}-images-idx3-ubyte.gz".format(kind))
    try:
        with gzip.open(labels_path, "rb") as lbpath:
            # Load labels data into a numpy array
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(images_path, "rb") as imgpath:
            # Load images data into a numpy array and reshape it
            # Each image is originally 28x28 pixels, so they are flattened into a 1D array of 784 elements
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        # Filter out images with labels not in targets
        mask = np.isin(labels, targets)
        images = images[mask]
        labels = labels[mask]
        #convert labels to binary
        labels = (labels == targets[0]).astype(int)

        return images, labels
    except FileNotFoundError:
        print(f"Error: Dataset files for {kind} not found in {path}. Please run download_mnist to fetch the dataset.")

def plot_test_predictions(data, predictions, target):
    '''Plot 10 of the test data along with their predictions and target labels.

    Args:
        data (np.ndarray): Test data.
        predictions (np.ndarray): Predictions of the model.
        target (np.ndarray): Target labels.
        n_samples (int, optional): Number of samples to plot. Defaults to 10.
    '''
    # Get n_samples random indices
    indices = np.random.choice(data.shape[0], 10, replace=False)

    # Plot the images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("Test Predictions", fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[indices[i]].reshape(28, 28), cmap="binary")
        ax.set(title=f"Label: {target[indices[i]]}, Prediction: {predictions[indices[i],0]:.2f}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    #Load MNIST data
    download_mnist("data")
    targets = [0, 1]
    X_train, y_train = load_mnist_binary("data", kind="train", targets=targets)
    X_test, y_test = load_mnist_binary("data", kind="test", targets=targets)

    #Set random seed for reproducibility
    np.random.seed(42)

    #Set hyperparameters
    learning_rate = 0.1
    epochs = 250
    batch_size = 1000

    #Prepare the data by scaling the input features
    X_train = X_train / 255
    X_test = X_test / 255

    #Initialize model
    model = Perceptron(input_size=X_train.shape[1], activation=SigmoidActivation(), optimizer=SGD(learning_rate))

    #Initialize model trainer
    model_trainer = ModelTrainer(model, loss_object=BinaryCrossEntropyLoss())

    #Train model
    trained_model = model_trainer.train_model(X_train, y_train, X_test, y_test, epochs, batch_size)

    #Plot test predictions
    test_predictions = trained_model.forward(X_test)
    plot_test_predictions(X_test, test_predictions, y_test)


