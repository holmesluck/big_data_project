#!/usr/bin/python
import numpy
from sklearn import preprocessing

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### 2. your code goes here


    # using the preprocessing method robust_scale to do the scale
    # and its reference is sklearn.preprocessing.robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
    # ages_scaled = preprocessing.RobustScaler(ages,quantile_range=(10,100))
    # # ages_scaled_array =ages_scaled.transform(ages)
    # # ages_scaled = numpy.array (ages_scaled)
    # net_worths = numpy.array(net_worths)
    # net_worths_scaled = preprocessing.RobustScaler (net_worths,quantile_range=(10,100))
    # predictions = numpy.array(predictions)
    # predictions_scaled = preprocessing.RobustScaler(predictions,quantile_range=(10,100))
    # print (ages)
    # print (len(ages))
    # ages_scaled = preprocessing.RobustScaler(quantile_range=(9,99))
    # ages_scaled_array = ages_scaled.fit_transform(ages)
    # print (ages_scaled_array)
    # print (len(ages_scaled_array))
    # print(ages)
    # print (len(ages))
    # print(predictions)
    # print(len(predictions))
    # print(net_worths)
    # print (len(net_worths))
    errors = (net_worths-predictions)**2
    cleaned_data = zip(ages,net_worths,errors)
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2][0],reverse=True)
    limit = int(len(net_worths)*0.1)

    print ("the list of tuples named cleaned_data is",cleaned_data[limit:])
    print ("the final number of the point is",len(cleaned_data[limit:]))
    return cleaned_data[limit:]

