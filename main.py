#Import the nessesary modules
import csv

class Area():
  """A class to calculate the area underneatha curve"""
  
  def __init__(self, start, end, watts):
    """Initiate any objects in use"""
    self.start = int(start)
    self.end = int(end) 
    self.watts = watts

  def calarea(self):
    """A class that calculates the area under a curve"""
    #Create a variable to store the final area
    areafin = 0
    #hardcoded constant time interval
    h = 1
    #Itirate for the given range
    while ((self.start) < (self.end)):
      #Calculate the area: trapezoid rule
      area = ((self.watts[self.start] + self.watts[self.start+1])*h)/2
      #Add the area to the total
      areafin = area + areafin
      #Increment 
      self.start = self.start + 1
    #Print the result
    print(areafin)

  def readfile(self):
    """A class to intake all of the information from the csv file"""
    #Open and process the file information
    file = "BatterySet2Offical.csv"
    with open(file) as f:
      reader = csv.reader(f)
      header_row = next(reader)
      #Create and store information
      self.watts = []
      for row in reader:
        self.watts.append(float(row[4]))
    #print(self.watts)

  def execute(self):
    """A class to call all functions in the respective order."""
    Area.readfile(self)
    Area.calarea(self)

while (True):
  #Determine the range that needs to be processed
  start = int(input("\nWhat is the start of the range? "))
  end = int(input("What is the end of the range? "))
  #Execute the operations
  Area(start, end, []).execute()