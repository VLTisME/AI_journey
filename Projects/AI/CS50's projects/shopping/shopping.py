import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    months = {"Jan" : 0, "Feb" : 1, "Mar" : 2, "Apr" : 3, "May" : 4, "June" : 5, "Jul" : 6, "Aug" : 7, "Sep" : 8, "Oct" : 9, "Nov" : 10, "Dec" : 11}
    visitorType = {"Returning_Visitor" : 1, "New_Visitor" : 0, "Other" : 0}
    bools = {"TRUE" : 1, "FALSE" : 0}

    evidence = []
    labels = []

    with open(filename, newline='') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=',')
        print('Loading Data from the CSV file... VLTisME')
        cur_line = 0
        for row in csvreader:
            cur_line += 1
            cur_line_evidence = []

            cur_line_evidence.append(int(row['Administrative']))
            cur_line_evidence.append(float(row['Administrative_Duration']))
            cur_line_evidence.append(int(row['Informational']))
            cur_line_evidence.append(float(row['Informational_Duration']))
            cur_line_evidence.append(int(row['ProductRelated']))
            cur_line_evidence.append(float(row['ProductRelated_Duration']))
            cur_line_evidence.append(float(row['BounceRates']))
            cur_line_evidence.append(float(row['ExitRates']))
            cur_line_evidence.append(float(row['PageValues']))
            cur_line_evidence.append(float(row['SpecialDay']))
            cur_line_evidence.append(months[row['Month']])
            cur_line_evidence.append(int(row['OperatingSystems']))
            cur_line_evidence.append(int(row['Browser']))
            cur_line_evidence.append(int(row['Region']))
            cur_line_evidence.append(int(row['TrafficType']))
            cur_line_evidence.append(visitorType[row['VisitorType']])
            cur_line_evidence.append(bools[row['Weekend']])
            
            evidence.append(cur_line_evidence)
            labels.append(bools[row['Revenue']])
    
    if len(labels) != len(evidence):
        sys.exit("Error: length of evidence and labels do not match, there might be corruption during loading data")    
    print('Data Loaded Successfully')

    return evidence, labels



def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    NumBuy = NumNotBuy = 0
    NumBuyCorrect = NumNotBuyCorrect = 0

    for i in range(len(labels)):
        if labels[i] == 1:
            NumBuy += 1
        else:
            NumNotBuy += 1
        if predictions[i] == 1 and labels[i] == 1:
            NumBuyCorrect += 1
        elif predictions[i] == 0 and labels[i] == 0:
            NumNotBuyCorrect += 1
    
    sensitivity = NumBuyCorrect / NumBuy
    specificity = NumNotBuyCorrect / NumNotBuy

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
