# code for portfolio and asset classes

import numpy as np
from scipy.optimize import minimize
#from sklearn import datasets, linear_model
#import matplotlib.pyplot as plt
import quandl
import math
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
import dill

authtoken = "JqvQjVgJ5iSqKswfJ82M"
MonthsImported = 24
NrMonthsPredicted = 12 ## This will be just an initial value.  Might be changed later.
NorMonthsPredModelTrain = 12
NumMonthsMonthsPreT0Fit = 0 #  The nr of months before now (T0) to stop the fit and aim the fit at.  From that time to T0 the 
                        #portfolio performance will be calculated and shown.
PortfolioRiskTolleranceList = [0, -0.1, 0.01, 0.1, 0.2, 0.3, 0.4] # 171015-this list is still good just making short list of 
# one at the moment in order to get the past compare stuff working.  171004:0.2 is giving great results        0.55 #0.875 #0.75 1.0 0.5    
#PortfolioRiskTolleranceList = [0, 0.1]
WindowMovingAveGrowth = 24 #months
#WindowMovingAveRisk = 24 #months
Epsilon = 0.00001 #small addition used to calc jacobian
NrDecimalsWeights = 4 #How many decimals to have in the weigths that are calculated.
NrClustersRiskReturn = 6 #Default nr of clusters that will be used for the unsupervised learning of the
                          #risk return space.
MinimumWeightpAsset = 0. #The minimum weight to be given too each asset in the optimization routine
NrNeuronsH1GrowthR = 100
NrNeuronsH2GrowthR = 10

assetdetails = [['S&P500', "LSE/CSPX", 0.9,  40.7128, -74.0059], #0
                ['Poland WIG',"WARSAWSE/WINDICES", 0.0,  52.2297, 21.0122],
                ['Brazil BVSP',"LSE/XMBR", 0.0,  -23.5505, -46.6333], #26
                ['Malta stock exchange',"MALTASE/INDEX",0.0, 35.9375, 14.3754], #32
                ['China SSEA',"LSE/XCHA",0.0,  39.9042, 116.4074], #1
                ['Japan Nikkei 225',"TSE/1321",0.0,  35.6895, 139.6917], #2
                ['Britain FTSE 100',"LSE/S100",0.0,  51.5074, -0.1278], #3
                ['Canada S&P TSX',"WFE/INDEXES_TMXGROUPSPTSXCOMPOSITE",0.0, 43.6532, -79.3832], #4
                ['France CAC 40',"LSE/CACX",0.0,  48.8566, 2.3522], #5
                ['Germany DAX',"BUNDESBANK/BBK01_WU3140",0.0 , 52.5200, 13.4050], #6
                ['Greece Athex Comp',"WFE/INDEXES_ATHENSSEGENERALPRICE",0.0,  37.9838, 23.7275], #7
                ['Italy FTSE/MIB',"LSE/IMIB",0.0,  41.9028, 12.4964], 
                ['Netherlands AEX',"LSE/IAEX",0.0,  52.3702, 4.8952],
                ['Spain Madrid SE',"LSE/SESP",0.0,  40.4168, 3.7038],
                ['Czech republic PX',"PRAGUESE/PX",0.0,  50.0755, 14.4378],
                ['Denmark OMXCB',"NASDAQOMX/OMXCBPI",0.0,  55.6761, 12.5683],
                ['Hungary BUX',"WFE/INDEXES_BUDAPESTSEBUMIX",0.0,  47.4979, 19.0402],
                ['Norway OSEAX',"WFE/INDEXES_OSLOBORSOSEBXPR",0.0,  59.9139, 10.7522],
                ['Sweden OMXS30',"NASDAQOMX/OMXS30",0.0,  59.3293, 18.0686],
                ['Switzerland SMI',"WFE/INDEXES_SIXSWISSEXCHANGESMI",0.0,  47.3769, 8.5417],
                ['Turkey BIST',"WFE/INDEXES_ISTANBULSEBIST",0.0, 41.0082, 28.9784], #18
                ['Australia All Ord.',"WFE/INDEXES_AUSTRALIANSECURITIESEXCHANGEALLORDINARYPRICE",0.0,  -33.8688, 140], #151.2093], #19
                ['Hong Kong Hang Seng',"NSE/HNGSNGBEES",0.0,  22.3964, 114.1095], #20
                ['India BSE',"BSE/BSE500", 0.0,  28.7041, 77.1025], #21
                ['Indonesia JSX',"WFE/INDEXES_INDONESIASEJSXCOMPOSITEINDEX",0.0, -6.1751, 106.8650], #22
                ['Malaysia KLSE',"WFE/MKTCAP_BURSAMALAYSIA",0.0, 3.1390, 101.6869], #23
                ['Pakistan KSE',"LSE/XBAK", 0.0, 24.8615, 67.0099], #24
                ['South Korea KOSPI',"WFE/INDEXES_KOREAEXCHANGEKOSPI", 0.0,  35.9078, 127.7669], #25
                ['Chile IGPA',"WFE/INDEXES_SANTIAGOSEIGPA",0.0,  -33.4489, -70.6693], #27  33.4489° S, 70.6693° W
                ['Colombia IGPC',"WFE/INDEXES_COLOMBIASECOLEQTY",0.0, 4.5709, -74.2973], #28
                ['Mexico IPC',"WFE/INDEXES_MEXICANASEIPCCOMPMX",0.0, 19.4326, -99.1332], #29
                ['Saudi Arabia Tadawul',"WFE/INDEXES_SAUDISETASI",0.0,  24.7136, 46.6753], #30
                ['Qatar All Share',"WFE/INDEXES_QATARSEQEALLSHAREINDEX",0.0,  25.2854, 51.5310], #31
                ['Zagreb Croatia',"ZAGREBSE/CROBEX",0.0, 45.8150, 15.9819], #33
                ['Belgrade Serbia SE',"BELGRADESE/BELEXLINE",0.0,  44.7866, 20.4489],  #34
                ['Ljubljana Slovenia SE',"LJUBSE/INDEX",0.0, 46.0569, 14.5058], #35
                ['Ukrainian Stock Index',"UKRSE/UX",0.0, 50.4501, 30.5234], #36
                ['Amman Jordan SE',"AMMANSE/INDEX_GENERAL_INDEX",0.0,  31.9454, 35.9284],  #37
                ['Tunis Tunisia SE',"TUNISSE/TUNINDEX",0.0, 33.8869, 9.5375]]  #38
#[,,1]]               


#RangeIncluded = [0, 6]
RangeIncluded = [0, len(assetdetails)]


#marketdata class definition -----------------------------------------------------------------------------

class marketdata: #for one asset
    def __init__(self, aname, aquandlcode, lat, lon):
        self.name = aname
        self.lat = lat
        self.lon = lon
        print('\n' + self.name)
        self.quandlcode = aquandlcode
        self.inittext = ''
        self.inittext+= '\nname : ' + self.name + ';    quandlcode : ' + self.quandlcode
        i = 0
        if i == 0:
            quandldf = quandl.get(self.quandlcode, authtoken="JqvQjVgJ5iSqKswfJ82M", collapse="monthly")
            if quandldf.columns[0] == 'Price':
                columnname = 'Price'
            elif quandldf.columns[0] == 'Open':
                columnname = 'Open'
            elif quandldf.columns[0] ==  'Value':
                columnname = 'Value'
            elif quandldf.columns[0] ==  'Index':
                columnname = 'Index'
            elif quandldf.columns[0] ==  'Index Value':
                columnname = 'Index Value'
            elif quandldf.columns[0] ==  'BET':
                columnname = 'BET'
            elif quandldf.columns[0] ==  'Close':
                columnname = 'Close'
            elif quandldf.columns[3] ==  'Index':
                columnname = 'Index'
            else:
                columnname = 'WIG20'
        pricehistorytemp = quandldf[columnname].values
        
        #print(self.lenpriceshitory)
        #self.pricehistory = self.pricehistory[-MonthsImported:]
        
        pricehistorytemp2 = list()
        for i in range(len(pricehistorytemp)):
            if math.isnan(pricehistorytemp[i]) == False:
                pricehistorytemp2.append(pricehistorytemp[i])
        self.pricehistory = np.array(pricehistorytemp2)
        self.lenpricehistory = len(self.pricehistory)

        benchmark = self.pricehistory[0]
        for i in range(self.lenpricehistory):
            self.pricehistory[i] = self.pricehistory[i]/benchmark
        #print('self.pricehistory:',self.pricehistory)

#asset class definition --------------------------------------------------------------------------

class asset:
    def __init__(self, aname, lat, lon, amarketdata):
        self.name = aname
        #print('\n' + self.name)
        
        self.pricehistory = amarketdata.pricehistory[:(amarketdata.lenpricehistory - NumMonthsMonthsPreT0Fit)]
        self.lenpricehistory = len(self.pricehistory)
        
        self.lengrowthpyearhistory = self.lenpricehistory - 12
        #if self.lenpricehistory%12 > 0:
 #           self.lengrowthpyearhistory += 1
        #print(self.lengrowthpyearhistory)  
        if self.lengrowthpyearhistory <= 0:
            self.growthpyearhistory = np.zeros(1)
            self.pricehistory = np.ones(1)
        else:
            self.growthpyearhistory = np.zeros(self.lengrowthpyearhistory)
        self.growthpyearmean = 0 #to be initialised

        self.lat = lat
        self.lon = lon

        self.fittedprediction = np.zeros((NrMonthsPredicted,1))
        self.calcgrowthrate()
        
        self.inittext = self.name + '; ' + 'self.growthrate : ' + str(self.growthrate)
        self.std = self.pricehistory.std()
        
        #print('self.std : ',self.std)
        #print('self.growthpyearstd : ',self.growthpyearstd)
        self.inittext += '; ' + 'self.std : ' + str(self.std)


    def calcgrothratetimeseries(self, oneseries): #calc the rate for a short series based on start and end
        #print('oneseries:',oneseries)
        thegrowthrate = ((oneseries[-1]/oneseries[0])**(1.0/float(len(oneseries))))**12.0 - 1.0
        #print('thegrowthrate:',thegrowthrate)
        return thegrowthrate

    def weightedave(self):
        growthrateseries = self.growthpyearhistory
        lengrowthrateseries = len(growthrateseries)
        if lengrowthrateseries < WindowMovingAveGrowth:
            lentoiterate = lengrowthrateseries
        else:
            lentoiterate = WindowMovingAveGrowth
        if lengrowthrateseries == 0:
            self.growthpyearmean = 0
            self.growthpyearstd = 0
        else:
            den = 0
            num = 0
            for i in range(lentoiterate):
                num += float(i + 1)/float(lentoiterate)*growthrateseries[lengrowthrateseries - lentoiterate + i]
                den += float(i + 1)/float(lentoiterate)
            self.growthpyearmean = num/den
            self.growthpyearstd = self.growthpyearhistory[-lentoiterate:].std()
            #print(self.name)
            #print(self.growthpyearmean)
            #print(self.growthpyearstd)
            #print('\n')


    def calcgrowthrate(self):
        #print('self.pricehistory[-1] :',self.pricehistory[-1])
        for month in range(0, self.lengrowthpyearhistory):
            endimonthseries = self.lenpricehistory - month
            if month == self.lengrowthpyearhistory - 1:
                startimonthseries = 0
            else:
                startimonthseries = self.lenpricehistory - month - 12
            self.growthpyearhistory[self.lengrowthpyearhistory - month - 1] = \
                    self.calcgrothratetimeseries(self.pricehistory[startimonthseries:endimonthseries])
        print('self.growthpyearhistory: ',self.growthpyearhistory)
        #self.growthpyearmean = self.growthpyearhistory.mean()
        self.weightedave()
        self.growthrate = (self.pricehistory[-1]**(1.0/float(self.lenpricehistory)))**12.0 - 1.0
        #print('self.growthrate : ',self.growthrate)
        #print('self.growthpyearmean: ',self.growthpyearmean)


#subportfolio class definition --------------------------------------------------------------------
class subportfolio:
    def __init__(self, assets, arisktolerrance):
        self.weights = list()
        self.assets = assets
        self.portfoliorisktollerance = arisktolerrance
        self.dbaldw = list()    #the derivative of the risk return balance wrt weights
        self.portfolioreturn = 0.0
        self.portfoliovariance = 0.0
        self.covmatrix = list()

    def initsubportfolio():
        pass

    def calcportfolioreturn(self):
        self.portfolioreturn = 0.0
        for i in range(len(self.assets)):
            self.portfolioreturn += \
               self.weights[i]*self.assets[i].growthpyearmean  
            #self.portfolioreturn += self.weights[i]*self.assets[i].growthrate 
        #print('\nself.portfolioreturn : ',self.portfolioreturn)


    def calccovmatrix(self):
        self.portfoliovariance = 0.0
        self.covmatrix = list()
        for i in range(len(self.assets)):
            matrixrow = list()
            for j in range(len(self.assets)):
                usestd = True
                if usestd == True and i == j:
                    coeff = (self.assets[j].growthpyearstd)**2
                else:
                    if usestd == True:

                        if self.assets[i].lengrowthpyearhistory >= self.assets[j].lengrowthpyearhistory:
                            lentouse = self.assets[j].lengrowthpyearhistory
                        else:
                            lentouse = self.assets[i].lengrowthpyearhistory
                        if lentouse > WindowMovingAveGrowth:
                            lentuse = WindowMovingAveGrowth
                            #print(lentouse)
                            #print(i,j)
                            #print(self.assets[i].growthpyearhistory[-lentouse:])
                            #print(self.assets[j].growthpyearhistory[-lentouse:])
                        if lentouse > 1:
                            coeff = np.cov(self.assets[i].growthpyearhistory[-lentouse:], \
                                   self.assets[j].growthpyearhistory[-lentouse:])[0][1]
                        else:
                            coeff = 0.0
                    else:
                        if self.assets[i].lenpricehistory >= self.assets[j].lenpricehistory:
                            lentouse = self.assets[j].lenpricehistory
                        else:
                            lentouse = self.assets[i].lenpricehistory
                        coeff = np.cov(self.assets[i].pricehistory[-lentouse:], \
                                   self.assets[j].pricehistory[-lentouse:])[0][1]
                        

                #print('i:',str(i),' j:',str(j),' coeff:',str(coeff))
                matrixrow.append(coeff)
                self.portfoliovariance += \
                    self.weights[i]*self.weights[j]*coeff
            self.covmatrix.append(matrixrow)
        #print('self.portfoliovariance : ',self.portfoliovariance)          
        #self.portfoliostd = math.sqrt(self.portfoliovariance)
        #print('self.portfoliostd : ',self.portfoliostd)

    def calcportriskreturnbalance(self):
        self.calcportfolioreturn()
        self.calccovmatrix()
        #self.portriskreturnbalance = self.portfoliostd - PortfolioRiskTollerance*self.portfolioreturn
        self.portriskreturnbalance = \
           self.portfoliovariance - \
            self.portfoliorisktollerance*self.portfolioreturn
        #print('self.portriskreturnbalance : ',self.portriskreturnbalance)

    def funcriskreturn(self, x, sign=1.0):
        self.weights = x
        self.calcportriskreturnbalance()
        return self.portriskreturnbalance

    def funcriskreturnderiv(self, x, sign=1.0):
        self.weights = x
        self.calcportriskreturnbalance()
        J0 = self.portriskreturnbalance
        for i in range(len(self.assets)):
            oldweight = self.weights[i]
            self.weights[i] += Epsilon
            self.calcportriskreturnbalance()
            J1 = self.portriskreturnbalance
            self.dbaldw[i] = (J1 - J0)/Epsilon
            self.weights[i] = oldweight
        return np.array(self.dbaldw)


    def optimiseportfolio(self): 
        cons = ({'type': 'eq', \
            'fun' : lambda x: np.array(sum(x) - 1), \
            'jac' : lambda x: np.ones(len(self.assets))})
        bnds = [(MinimumWeightpAsset, None)] #was [(0, None)]
        for i in range(1, len(self.assets)):
            bnds.append((0, None))
        #print(bnds)
        #print(self.weights)
        res = minimize(self.funcriskreturn, self.weights, args=(), \
                       jac=self.funcriskreturnderiv, \
                       bounds=bnds, constraints=cons, method='SLSQP', options={'disp': True})
        for i in range(len(self.assets)):
            self.weights[i] = res.x[i]
            
        
        
#portfolio class definition -----------------------------------------------------------------------------------------
       
class portfolio:  #Over arching class for the entire model layer of the app.  Presentation layer is in
                    #app.py
    def __init__(self):
        self.nrassets = RangeIncluded[1] - RangeIncluded[0]
        #print(self.nrassets)
        self.marketdatas = list()
        self.assets = list()
        self.subportfolios = list()
        self.nrplots = 0
        self.terminaltext = ''; #String that will be output to the terminal on the html when needed.
        self.wouldbetrendlist = list()


    def initportfolio(self, datalocal = False):
        
        j = 0
        
        self.riskreturnlabels = list()
        
        if datalocal == False:
            for i in range(RangeIncluded[0], RangeIncluded[1]):
                self.marketdatas.append(marketdata(assetdetails[i][0], assetdetails[i][1], assetdetails[i][3], \
                                     assetdetails[i][4]))
        else:
            self.marketdatas = dill.load(open('portfoliobackup.pkd', 'rb'))
            #self.weights.append(1.0/float(self.nrassets))
            
            #print('self.weights ',j,' : ',self.weights)
            j += 1
        #If the API worked, then one should at this point be able to write the price histories to disk to enable
            #easier getting the data next time.
         
        if datalocal == False:
            print('Dilling portfolio to disk...')
            dill.dump(self.marketdatas, open('portfoliobackup.pkd', 'wb'))

        if datalocal == True:
            self.initportfoliometrics()

        

    def initportfoliometrics(self):
        for i in range(self.nrassets):
            self.assets.append(asset(self.marketdatas[i].name, self.marketdatas[i].lat, self.marketdatas[i].lon, self.marketdatas[i]))

        for j in range(len(PortfolioRiskTolleranceList)):
            self.subportfolios.append(subportfolio(self.assets, PortfolioRiskTolleranceList[j]))
            for i in range(RangeIncluded[0], RangeIncluded[1]):
                self.subportfolios[j].dbaldw.append(1.0) #just an initial value
                self.subportfolios[j].weights.append(assetdetails[i][2])
        
            self.subportfolios[j].calcportfolioreturn()
            self.subportfolios[j].calccovmatrix()
            self.subportfolios[j].calcportriskreturnbalance()
        #self.printcovmatrix()
    

    def getportfolioslocal(self):
        self.subportfolios = dill.load(open('subportfoliosbackup.pkd', 'rb'))

    def printtoterminal(self, s):
        self.terminaltext += s + '\n'
         

    def predictreturns(self): #Use neural network to predict asset returns
        mlp = MLPRegressor(hidden_layer_sizes=(NrNeuronsH1GrowthR, NrNeuronsH2GrowthR))
        for i in range(self.nrassets):
            #print(halfph)
            lendata = self.assets[i].lenpricehistory - NrMonthsPredicted
            inputs = np.array(self.assets[i].pricehistory[0:lendata]).\
                              reshape((lendata,1))
            outputactuals = np.ravel(np.array(self.assets[i].pricehistory[NrMonthsPredicted:]).\
                              reshape((lendata,1)))
            mlp.fit(inputs, outputactuals)
            self.assets[i].fittedprediction = mlp.predict(\
                np.array(self.assets[i].pricehistory).reshape((self.assets[i].lenpricehistory,1)))
            

    def unsupervisedkmeansonriskreturn(self):
        km = KMeans(n_clusters = NrClustersRiskReturn)
        km.fit([[asset.growthpyearmean, asset.growthpyearstd] for asset in self.assets])
        self.riskreturnlabels = list()
        for i in range(self.nrassets):
            self.riskreturnlabels.append(km.labels_[i])
        print(km.labels_)
            
    

    def printportstats(self, portindex):
        s = ''
        self.printtoterminal('Portfolio nr:\t\t' + str(portindex))
        for i in range(self.nrassets):
            if self.subportfolios[portindex].weights[i] > 0:
                s = self.assets[i].name
                if len(s)//8 <= 1:
                    s += '\t\t'
                else:
                    s+= '\t'
                s += str(round(self.subportfolios[portindex].weights[i],NrDecimalsWeights)) + ' '
                self.printtoterminal(s)
        self.printtoterminal('\nPortfolio return:\t\t' + str(round(self.subportfolios[portindex].portfolioreturn,NrDecimalsWeights)))
        #self.printoterminal('Portfolio std: ' + str(self.portfoliostd))
        self.printtoterminal('Portfolio variance:\t\t' + str(round(self.subportfolios[portindex].portfoliovariance,NrDecimalsWeights)))
        self.printtoterminal('Risk Return Balance:\t\t' + str(self.subportfolios[portindex].portriskreturnbalance))
        self.printtoterminal('PortfolioRiskTollerance:\t\t' + str(PortfolioRiskTolleranceList[portindex]))
        self.printtoterminal('\n\n')


    def optimiseportfolio(self, optimise=True):
        self.printportstats(0)

        for i in range(1, len(PortfolioRiskTolleranceList)):
            #print('\n')
            if optimise == True:
                self.subportfolios[i].optimiseportfolio()
            #print('\n')
            if optimise == True:
                self.subportfolios[i].calcportriskreturnbalance()
            self.printportstats(i)

        if optimise == True:
            #print('Dilling sub-portfolios to disk...')
            dill.dump(self.subportfolios, open('subportfoliosbackup.pkd', 'wb')) 

    def howportwouldhavedone(self, portindex): #Follow the actual market figures NumMonthsMonthsPreT0Fit months into the future to T0 for plot.
        returnafterperiod = np.ones(NumMonthsMonthsPreT0Fit + 1)
        for period in range(NumMonthsMonthsPreT0Fit):
            fracincformonth = 0.0
            for i in range(self.nrassets):
                if self.marketdatas[i].lenpricehistory >= NumMonthsMonthsPreT0Fit:
                    marketvalueatend = self.marketdatas[i].pricehistory[-NumMonthsMonthsPreT0Fit + period]
                    marketvalueatstart = self.marketdatas[i].pricehistory[-NumMonthsMonthsPreT0Fit + period - 1]
                else:
                    marketvalueatstart = 1.0
                    marketvalueatend = 1.0
                fractionincinmonthforasset = (marketvalueatend - marketvalueatstart)/marketvalueatstart
                fracincformonth += self.subportfolios[portindex].weights[i]*fractionincinmonthforasset
            returnafterperiod[period + 1] = returnafterperiod[period]*(1 + fracincformonth)
        return returnafterperiod
            


        

 
