import csv

'''
sample
{
        "Class": "republican", 
        "handicapped-infants": "n", 
        "water-project-cost-sharing": "y", 
        "adoption-of-the-budget-resolution": "n", 
        "physician-fee-freeze": "y", 
        "el-salvador-aid": "y", 
        "religious-groups-in-schools": "y", 
        "anti-satellite-test-ban": "n", 
        "aid-to-nicaraguan-contras": "n", 
        "mx-missile": "n", 
        "immigration": "y", 
        "synfuels-corporation-cutback": "n", 
        "education-spending": "y", 
        "superfund-right-to-sue": "y", 
        "crime": "y", 
        "duty-free-exports": "?", 
        "export-administration-act-south-africa": "n"
    }
'''


def parse(filename):
    '''
    takes a filename and returns attribute information and all the data in array of dictionaries
    '''
    # initialize variables

    out = []
    csvfile = open(filename, 'r')
    fileToRead = csv.reader(csvfile)

    headers = next(fileToRead)

    # iterate through rows of actual data
    for row in fileToRead:
        out.append(dict(zip(headers, row)))
    return out
