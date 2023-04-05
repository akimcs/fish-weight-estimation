import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn import model_selection, linear_model, metrics

# read csv file into a dataframe.
df = pd.read_csv("fish_data.csv")

# grab columns from csv file.
len_data = df.values[:, 2:7]
wgt_data = df.values[:, 1]

# split array data into train and test subsets.
len_train, len_test, wgt_train, wgt_test = model_selection.train_test_split(len_data, wgt_data, test_size=0.3)

# create a linear regression model using training data
lin_reg_model = linear_model.LinearRegression()
lin_reg_model.fit(len_train, wgt_train)


def predicting_application():
    """print UI. provide predicted weight from inputted dimensions."""
    print(r'       .')
    print(r'      ":\"')
    print(r'    ___:____     |"\/"|')
    print(r"  ,'        `.    \  /")
    print(r'  |  O        \___/  |')
    print(r'~^~^~^~^~^~^~^~^~^~^~^~^~')
    print("---------------------------------------")
    print("Welcome to TastyFish's fish weight prediction tool.\n"
          "Using the dimensions of a fish, our machine learning algorithm "
          "will predict its weight in grams (rounded to 2 decimal places).\n\n"
          "Enter the vertical length, diagonal length, cross length, height, and diagonal width (in centimeters).\n"
          "(Example: 23.2, 25.4, 30, 11.52, 4.02)\n\n"
          "*Type 'end' to return to main menu.*")

    while True:
        measurements = input("Input: ")
        if measurements == "end":
            main_application()
        else:
            try:
                measurements_list = list(map(float, measurements.split(", ")))
                if len(measurements_list) == 5:
                    print("Your fish weighs", round(lin_reg_model.predict([measurements_list])[0], 2), "gram(s)\n")
                else:
                    print("ERROR: Use 5 numbers only.\n")
            except ValueError:
                print("ERROR: Use 5 numerical inputs only, each separated by a ', '\n")


def visuals():
    """provide visuals and model accuracy data."""
    wgt_pred = lin_reg_model.predict(len_test)

    # r-squared value
    pred_accuracy = metrics.r2_score(wgt_test, wgt_pred)
    print("---------------------------------------\n")
    print("The R-squared value of the linear regression model is ", round(pred_accuracy, 2))

    # 3 visuals
    print("3 visualizations of data wil be shown (histogram, scatter plot, and prediction error of linear regression model.\n")
    prediction_error = metrics.PredictionErrorDisplay(y_true=wgt_test, y_pred=wgt_pred)
    prediction_error.plot()  # prediction error of regression model
    df.hist()  # histogram
    scatter_matrix(df)  # scatter plot
    pyplot.show()

    main_application()


def main_application():
    print("---------------------------------------")
    print("Welcome to TastyFish's new machine learning product.\n")
    print("Would you like to ('A') use our machine learning tool OR ('B') view the visuals of this project?")
    decision = input("Your Input ('A' or 'B' or 'end') Here: ")
    if decision == 'A':
        predicting_application()
    elif decision == 'B':
        visuals()
    elif decision == 'end':
        quit()
    else:
        print("ERROR: please input 'A' or 'B' or 'end'.")


main_application()
