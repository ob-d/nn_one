# nn_one
Experimental neural network project for crypto price prediction
---
---
---
---

## INSTALLATION
    To check if Python is installed Open a Terminal and enter 
    python --version
    If it does not work with your current version, you need to update to the latest stable version for compatibility with the libraries.
    
    To run the script, you'll need to have Python installed on your computer. You can download and install Python from the official website link below.

    Once you have Python installed, you'll also need to install the required libraries. You can do this by opening a command prompt or terminal window and running the following command:

## 1 first install python
```
https://www.python.org/downloads/
```
## 2 pip
```
pip --version
```
## 3 install libraries
```
pip install torch numpy pandas requests
```
## 4 open Vscode and start script
```
# choose one of the networks avilable and enter the command in your terminal
python nnBot.py
python nnSTM.py
```


This will execute the script using the Python interpreter. The script will fetch historical price data from the Binance API, preprocess the data, train a DFFNN on the data, and then print the loss at each epoch of training.

If everything is set up correctly, you should see output similar to this:

```
Epoch [1/100], Loss: 0.0001
Epoch [2/100], Loss: 0.0001
Epoch [3/100], Loss: 0.0001
...
```

You can stop the script at any time by pressing `Ctrl+C`.


# `numpy`
    LOSS FUNCTION OPTIMIZATION
    It's great to hear that the script is running without errors! However, it seems like the loss is quite high and not decreasing as expected. There are several things you can try to improve the performance of the model:

    1. **Normalize the data**: Neural networks often perform better when the input data is normalized to have zero mean and unit variance. You can normalize your data using a library such as `scikit-learn` or by manually subtracting the mean and dividing by the standard deviation.

    2. **Tune the hyperparameters**: The performance of a neural network can depend on the values of its hyperparameters, such as the number of hidden neurons, the learning rate, and the number of training epochs. You can try experimenting with different values of these hyperparameters to see if they improve the performance of your model.

    3. **Try a different network architecture**: The DFFNN used in the script is a simple architecture that may not be well-suited for all tasks. You can try using a different network architecture, such as a recurrent neural network (RNN) or long short-term memory (LSTM) network, which are better suited for handling sequential data.

    4. **Use more data**: Neural networks often perform better when trained on larger amounts of data. You can try increasing the `limit` parameter in the Binance API request to fetch more historical price data.

    I hope these suggestions help you improve the performance of your model. Let me know if you have any further questions or need any additional help.