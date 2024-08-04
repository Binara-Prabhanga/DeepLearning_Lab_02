# Deep Learning – Lab 2

## Instructions

For this lab, you can either use the Anaconda Python distribution or Google Colab.

To install the Anaconda Python distribution, visit: [Anaconda Individual Edition](https://www.anaconda.com/products/individual)

### Part 1: Backpropagation

1. **Upload `Backprop.ipynb` to Jupyter Notebook (or Google Colab)**
   - Review and understand the code.
   - Increase the number of iterations (epochs) and observe if the prediction accuracy improves.
   - **Note**: You may need to copy `image.png` to the home directory.

### Part 2: Neural Network Sample

2. **Upload `NN_sample.ipynb` to Jupyter Notebook (or Google Colab)**
   - Review and understand the code.
   - Add the following text and code cells to the notebook and run it again.

#### Text Cell:
```
Now, let's try out several hidden layer sizes.
4.6 - Tuning hidden layer size (optional/ungraded exercise)
```

#### Code Cell:
```python
# This may take about 2 minutes to run
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
```

- **Questions:**
  1. What happens when the number of hidden nodes increase?
  2. Can you explain the pattern of the accuracy when the hidden nodes increase?

- **Note**: Copy `planar_utils.py` and `testCases.py` to the home directory.

### Part 3: MLP with MNIST Dataset

3. **Run `MLP_with_MNIST_dataset.ipynb` using Jupyter Notebook (or Google Colab)**
   - Review and understand the code.
   - Improve the test accuracy of the model by changing the hyperparameters.
   - Add L1 and L2 regularization terms to the model and retrain it.
   - Visualize class-wise test dataset performance using a confusion matrix.

### Part 4: Optional Exercise

4. **Neural Network Playground (Optional/No need to submit anything)**
   - Visit the [Neural Network Playground](https://playground.tensorflow.org/).
   - Experiment with different hyperparameters.
   - Explore using L1 and L2 regularization to reduce overfitting.

## Submission Instructions

1. **Create a GitHub repository for the Lab 2 submission.**

2. **Export and upload the following modified notebooks to the repository:**
   - The modified notebook used in Exercise 1 (`Backprop.ipynb`).
   - The modified notebook used in Exercise 2 (`NN_sample.ipynb`).
   - The modified notebook used in Exercise 3 (`MLP_with_MNIST_dataset.ipynb`).

3. **Answer the questions from Exercise 2 in a text file.** 
   - Include this text file in the repository.

4. **Create a text file with your registration number as the file name.**
   - Add the repository link to this text file.
   - Submit this text file to the Lab 2 submission link.

5. **Ensure results are visible in the notebooks.**

## Lab 2 Repository Structure

```markdown
.
├── Backprop.ipynb
├── NN_sample.ipynb
├── MLP_with_MNIST_dataset.ipynb
├── Exercise_2_Answers.txt
└── [Your_Registration_Number].txt
```

**Note:** Make sure all required files (`image.png`, `planar_utils.py`, `testCases.py`) are included in the appropriate directories if needed for the notebooks to run correctly.
