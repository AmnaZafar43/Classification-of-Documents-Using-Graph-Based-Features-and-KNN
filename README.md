# Classification-of-Documents-Using-Graph-Based-Features-and-KNN

###Developers Name

Amna Zafar(2021-CS-27)
Amna Salman(2021-CS-23)

###***Project Name:***		

Classification-of-Documents-Using-Graph-Based-Features-and-KNN

###***Project Description:***

There are three types of categories for which we are asked to scrapp data. The categories are 'Fashion and Beauty', 'Sports' and 'Science & Education'.We scrapped 15 documents of each topic and each document contains 500 words. Then after successful scrapping we do preprocessing(tokenization etc) and represent each document(containing 500 words) as directed graph.Then calculate graph distance, apply KNN. We then prepare the training data and train our model, then give it test data for prediction. We then evaluate the prediction,calculate accuracy,F1 score, Jaccard similarity and plot our confusion matrix. Our accuracy and related scores are 100.00% accurate which means our model is perfectly trained.

###*How to run project locally*
There are two ways: By cloning github repository or extract zip file.

1. **Clone Github repository:**Open repository link can show you a green bbutton of code click on that and copy the given `http` link. Now open any editor and write `git clone "paste copied link here"` and complete folder clones and you now can run F5 or run icon shown on editor to run project.

2. **Extract Zip File:**Extract zip file in anywhere on your desktop and run F5 or run icon shown on editor like vs code to run the project.

###*Requirements for running code sucessfully*
The requirements that you have to check before running project is:

1. **Python:**Make sure you have python installed to your system. In case, if it is not installed click on link to install `https://www.python.org/downloads/`.

2. **Libraries:**Run the following commands in your terminal for running code properly:
    * `pip install pandas`
    * `pip install networkx`
    * `pip install matplotlib`
    * `pip install seaborn`
    * `pip install sklearn`
Make sure you run these commands in your command propmt, otherwise code will not run and it give error. 