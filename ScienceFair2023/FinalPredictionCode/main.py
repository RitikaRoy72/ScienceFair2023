#Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


class MachineLearning():
  
  def __init__(self, df, dfCap, X_df, y, X_train, Y_train, X_test, Y_test):
    """Initiate all objects"""
    self.df = df
    self.dfCap = dfCap
    self.X_df = X_df
    self.y = y
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
  

  def OriginalGraph(self):
    """Graph the original data"""
    #Read the data frame, 
    # use the given to append or remove info
    self.df = pd.read_csv('BatterySetAnalysis.csv')
    self.df["capacity"] = self.df.Area/60.0
    self.dfCap = self.df.drop("Area", axis = 1)
    #Initiate the suplot, read it the data
    fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"})
    surf = ax.scatter(self.df.Iterations, self.df.Temperature, self.df.capacity, s=15)
    #Set te labels and plot
    ax.set_xlabel("Time (Cycles)", fontsize=15)
    ax.set_ylabel("Temperature ($^\circ$F)", fontsize=15)
    ax.set_zlabel("Capacity (MiliWatts-Hours)", fontsize=11)

    ax.tick_params(axis='x',labelsize="7")
    ax.tick_params(axis='y',labelsize="7")
    ax.tick_params(axis='z',labelsize="7")
    
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.savefig("OriginalData.png")
    #plt.show()
    #Re-set the objects
    self.X_df = self.dfCap.drop('capacity', axis = 1)
    self.y = self.dfCap.capacity
  

  def ThreeDGraph(self, model, name):
    """A class to graph a 3-D graph"""
    #Get the predicted data from values
    #Stored in objects
    y_pred = model.predict(self.X_df)
    #Initiate the suplot and read the data
    fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"})
    surf = ax.scatter(self.X_df.Iterations, self.X_df.Temperature, self.y, s = 15, c= 'red', label = "Original")
    surf1 = ax.scatter(self.X_df.Iterations, self.X_df.Temperature, y_pred, s = 15, c= 'blue', label = "Predicted")
    #Stylistic elements
    ax.set_xlabel("Cycles", fontsize=15)
    ax.set_ylabel("Temperature ($^\circ$F)", fontsize=15)
    ax.set_zlabel("Capacity (MilliWatt-Hours)", fontsize=11)

    ax.tick_params(axis='x',labelsize="7")
    ax.tick_params(axis='y',labelsize="7")
    ax.tick_params(axis='z',labelsize="7")
    
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.savefig(name)
    #plt.show()

  
  def TwoDGraph(self, model, name):
    #Get the predicted data fro values
    #stored in objects
    y_df_pred = model.predict(self.X_df)
    #Initiate the subplot and read the data
    fig, ax = plt.subplots()
    ax.scatter(self.X_df.Iterations, self.y, s=1000, c='red', label = "Original")
    ax.scatter(self.X_df.Iterations, y_df_pred, s=1000, c='blue', label = "Predicted")
    #Stylistic elements
    ax.legend(fontsize="100")
    
    ax.set_xlabel("Time (Cycles)", fontsize=150)
    ax.set_ylabel("Capacity (MilliWatts-Hours)", fontsize=150)

    ax.tick_params(axis='x',labelsize="100")
    ax.tick_params(axis='y',labelsize="100")
    
    fig.set_figwidth(60)
    fig.set_figheight(40)
    plt.savefig(name)
    #plt.show()

    
  def TrainModels(self):
    #Initiate the trianing objects
    #Save 20% data, use the rest for Machine Learn
    self.X_train, self. X_test, self.Y_train, self.Y_test = train_test_split(self.X_df, self.y, test_size=0.2, random_state=33)

  
  def LinearRegression(self):
    """Create a linear regression model"""
    #Initiate and train the linear regression
    lr = LinearRegression()
    lr_model = lr.fit(self.X_train, self.Y_train)
    print("Linear Regression")
    #Print the coefficents
    print(lr_model.coef_)
    print(lr_model.intercept_)
    y_pred = lr_model.predict(self.X_test)
    #Print the accuracy error
    print(metrics.r2_score(self.Y_test, y_pred))
    print(metrics.mean_squared_error(self.Y_test, y_pred))
    print("\n")
    y_df_pred = lr_model.predict(self.X_df)
    #Create and save the graphs
    MachineLearning.ThreeDGraph(self,  lr_model, "LinearPredicted3D.png")
    MachineLearning.TwoDGraph(self, lr_model, "LinearPredicted2D.png")

  
  def QuadraticRegression(self):
    """Create a quadratic regresion model"""
    #Initiate and train the quadratic regression
    poly2 = make_pipeline(PolynomialFeatures(2),LinearRegression())
    poly2.fit(self.X_train,self.Y_train)
    y_pred2 = poly2.predict(self.X_test)
    print("Quadratic Regression")
    #Print the coefficents
    print(poly2.steps[1][1].coef_)
    print(poly2.steps[1][1].intercept_)
    #Print the accuracy error
    print(metrics.r2_score(self.Y_test, y_pred2))
    print(metrics.mean_squared_error(self.Y_test, y_pred2))
    print("\n")
    #Create and save the graphs
    MachineLearning.ThreeDGraph(self, poly2, "QuadraticPredicted3D.png")
    MachineLearning.TwoDGraph(self, poly2, "QuadraticPredicted2D.png")

  
  def CubicRegression(self):
    """Initiate and train the cubic regression"""
    #Initiate and train the cubic regression
    poly3 = make_pipeline(PolynomialFeatures(3),LinearRegression())
    poly3.fit(self.X_train,self.Y_train)
    y_pred3 = poly3.predict(self.X_test)
    print("Cubic Regression")
    #Print the coefficents
    print(poly3.steps[1][1].coef_)
    print(poly3.steps[1][1].intercept_)
    #Print the accuracy errror
    print(metrics.r2_score(self.Y_test,y_pred3))
    print(metrics.mean_squared_error(self.Y_test, y_pred3))
    print("\n")
    #Create and save the graphs
    MachineLearning.ThreeDGraph(self, poly3, "CubicPredicted3D.png")
    MachineLearning.TwoDGraph(self, poly3, "CubicPredicted2D.png")


  def execute(self):
    """A method to execute the operations in the correct order"""
    MachineLearning.OriginalGraph(self)
    MachineLearning.TrainModels(self)
    MachineLearning.LinearRegression(self)
    MachineLearning.QuadraticRegression(self)
    MachineLearning.CubicRegression(self)
    

#Call the function
MachineLearning(0,0,0,0,0,0,0,0,).execute()