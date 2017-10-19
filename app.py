#Developed by Johannes Henning Viljoen

from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.palettes import viridis
from bokeh.models import (GMapPlot, GMapOptions, ColumnDataSource, \
                          Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool)
#from bokeh.charts import HeatMap, bins
from bokeh.models import DatetimeTickFormatter
from scipy.ndimage import imread
import math
import numpy as np
import portfolio as pf
import random
from random import randint

from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.layouts import row
import dill

#change 1

BasicURL1 = 'https://www.quandl.com/api/v3/datasets/WIKI/'
BasicURL2 = '.json'
par = {'column_index':'4','start_date':'2017-08-08','end_date':'2017-09-08','collapse':'daily','api_key':'JqvQjVgJ5iSqKswfJ82M'}
MonkeyText = ''
RiskReturnLogBias = 1.1     #log bias applied to get the size of circles on the plot correct.
RiskReturnPlotScaling = 0.5 #scaling applied to get the relative size of circles on the plot correct.
AssetChosenForDetails = 15 #Index of asset that will be studied in more detail in the diag for it.
RangeOfAssetsForPredictDiag = range(1,4)
SizeOfDistRiskReturnPlots = 570

#Initialise main class for total app ------------------------------------------------------------

app = Flask(__name__)
port = pf.portfolio()
port.initportfolio(datalocal = True)
port.subportfolios = dill.load(open('subportfoliosbackup.pkd', 'rb'))
port.wouldbetrendlist = dill.load(open('wouldbetrend.pkd', 'rb'))
portfolionumber = 0 #global variable for choosing different ports in the webpages.

#Utility functions
def percentage(afraction):
    return round(afraction*100,2)

#Web routines -----------------------------------------------------------------------------------

@app.route('/', methods = ['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('appindex.html', terminaltext = MonkeyText)

    #else:
  #      app.ticker = request.form['tickername']
 #       return redirect('/trendstock')

@app.route('/navigate', methods = ['GET','POST'])
def navigate():
    return redirect('/')



@app.route('/appdetails', methods = ['GET','POST']) #Detail app page
def appdetails(): 
    return render_template('appdetails.html', terminaltext = '')


@app.route('/aboutpage', methods = ['GET','POST']) #this function will trend the time series data for all assets
def aboutpage(): #remember the function name does not need to match the URL
    return render_template('about.html', terminaltext = '')


@app.route('/getdata', methods = ['GET','POST']) #this function will trend the time series data for all assets
def getdata(): #remember the function name does not need to match the URL
    port.initportfolio()
    return redirect('/plottrends')


@app.route('/getdatalocal', methods = ['GET','POST']) #this function get data locally
def getdatalocal(): 
    port.initportfolio(datalocal = True)
    return redirect('/plottrends')


@app.route('/getportfolioslocal', methods = ['GET','POST']) #this function will import saved portfs.
def getportfolioslocal():
    port.getportfolioslocal()
    return redirect('/')


@app.route('/plottrends', methods = ['GET','POST'])
def plottrends():
    x = range(pf.MonthsImported)
    yforplot = port.marketdatas[0].pricehistory

    #img = imread('worldmap-background-design_1127-2318.jpg')
    
    # create a new plot with a title and axis labels
    p = figure(title='Assets time series data.  Source: Quandl.com', \
               x_axis_label='x', y_axis_label='y')

    # add a line renderer with legend and line thickness
    mypalette = viridis(port.nrassets)
    #print(len(mypalette))
    for i in range(port.nrassets):
        #print(i)
        p.line(range(-port.marketdatas[i].lenpricehistory + 1,1), port.marketdatas[i].pricehistory*100.0, \
               legend=port.marketdatas[i].name, line_color = mypalette[i], \
               line_width=2)

    p.legend.location = "top_left"
    #p.height = 100
    #p.width = 100
    p.xaxis.axis_label = 'Time (months before T0)'
    p.yaxis.axis_label = 'Return (USD when each market index starts at USD 100)'
        
    plot = p
    script, div = components(plot)
    return render_template('plottrends.html', script = script, div = div, \
                           inittext='')

    
@app.route('/portfolioinvestdistribution', methods = ['GET','POST'])
def portfolioinvestdistribution():
    #if request.args['messages']:
    #    portfolionr = request.args['messages']
    #else:
    #    portfolionr = portfolionumber
    portfolionr = 0
    #if request.method == 'POST':
    #    portfolionr = int(request.form['pageinput'])
    if request.method == 'POST':
        if request.form['submit'] == 'Risk Level 0':
            portfolionr = 0
        elif request.form['submit'] == 'Risk Level 1':
            portfolionr = 1
        elif request.form['submit'] == 'Risk Level 2':
            portfolionr = 2
        elif request.form['submit'] == 'Risk Level 3':
            portfolionr = 3
        elif request.form['submit'] == 'Risk Level 4':
            portfolionr = 4
        elif request.form['submit'] == 'Risk Level 5':
            portfolionr = 5
        elif request.form['submit'] == 'Risk Level 6':
            portfolionr = 6

    text = 'Portfolio geographical distribution for portfolio nr. ' + str(portfolionr)
    
    GoogleAPIKey = 'AIzaSyAk7xl9SefWfefumGBJEHVdel0xDS2vZwI'
  
    map_options = GMapOptions(lat=0, lng=-0, map_type="roadmap", zoom=1)

    portdistrplot = GMapPlot(x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options)
    portdistrplot.title.text = "Portfolio geographic and capital distribution for portfolio nr. " + str(portfolionr)

    portdistrplot.api_key = GoogleAPIKey

    wlist = list()
    for i in range(port.nrassets):
        w = port.subportfolios[portfolionr].weights[i]
        if w > 0:
            w = 4 + math.sqrt(w)*35
        wlist.append(w)
    #math.log(1.2 + weight)*30 
    source2 = ColumnDataSource(data=dict(  \
        lat=[asset.lat for asset in port.assets], lon=[asset.lon for asset in port.assets],\
        radii = wlist))
    y = [asset.lat for asset in port.assets]
    x = [asset.lon for asset in port.assets]
    #size=15
    circle = Circle(x="lon", y="lat", size="radii", fill_color="blue", fill_alpha=0.8, line_color=None)
    #circle = Circle(x=x[0], y=y[0], size=15, fill_color="blue", fill_alpha=0.8, line_color=None)
    portdistrplot.add_glyph(source2, circle)

    portdistrplot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    portdistrplot.width = SizeOfDistRiskReturnPlots
    portdistrplot.height = SizeOfDistRiskReturnPlots
    
    
    #portfolio risk-return plot in this section
    text = 'Portfolio risk-return plot for portfolio nr. ' + str(portfolionr)
    port.unsupervisedkmeansonriskreturn()
    x = np.array([math.log10(asset.growthpyearstd + 1) for asset in port.assets]) #convert std to var
    y = np.array([asset.growthpyearmean*100.0 for asset in port.assets])
    portvar = port.subportfolios[portfolionr].portfoliovariance
    print(portvar)
    processedportvar = portvar/abs(portvar + 0.00001)*(abs(portvar)**0.5)
    x = np.append(x, math.log10(processedportvar + 1))

    y = np.append(y, port.subportfolios[portfolionr].portfolioreturn*100.0)
    labels = np.array(port.riskreturnlabels)
    #labels = np.append(labels,pf.NrClustersRiskReturn)
    print(labels)
    wlist = list()
    for i in range(port.nrassets):
        w = port.subportfolios[portfolionr].weights[i]
        if w > 0:
            w = 0.007 + math.sqrt(w)*0.04
        wlist.append(w)           
    radii = np.array(wlist)         
    print(radii)
    radii = np.append(radii,0.015) #for portfolio data point
    print(radii)
    colors = ["#%02x%02x%02x" % (int(r), int(g), 150) \
              for r, g in zip(50+200/((pf.NrClustersRiskReturn+1)-1)*labels, \
                                   30+200//((pf.NrClustersRiskReturn+1)-1)*labels)]
    colors.append("#%02x%02x%02x" % (200, 0, 0))
    #print(colors)
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    riskreturnplot = figure(tools = TOOLS)
    #plt.scatter(self.assets[i].std,self.assets[i].growthrate, color = colors[i], \
    #            label=lbl)
    riskreturnplot.scatter(x, y, radius = radii, fill_color = colors, fill_alpha = 0.6, line_color = None)
    riskreturnplot.title.text = text
    riskreturnplot.width = SizeOfDistRiskReturnPlots
    riskreturnplot.height = SizeOfDistRiskReturnPlots
    riskreturnplot.xaxis.axis_label = 'Estimated risk metric (log scale)'
    riskreturnplot.yaxis.axis_label = 'Estimated future return (perc. increase p.a.)'

    #Now the two plots will be put next to each other
    plot = row(portdistrplot, riskreturnplot)

    #plot.width = 600
    #plot.height = 300

    script, div = components(plot)
    if len(port.assets) > 0:
        nrassets = port.nrassets
    else:
        nrassets = 0

    return render_template('portfolioinvestdistribution.html', script = script, div = div)


@app.route('/portfolioriskreturnplot', methods = ['GET','POST'])
def portfolioriskreturnplot():
    portfolionr = 0
    if request.method == 'POST':
        portfolionr = int(request.form['pageinput'])
    text = 'Portfolio risk-return plot for portfolio nr. ' + str(portfolionr)
    port.unsupervisedkmeansonriskreturn()
    x = np.array([math.log10(asset.growthpyearstd) for asset in port.assets]) #convert std to var
    y = np.array([asset.growthpyearmean for asset in port.assets])
    x = np.append(x, math.log10(port.subportfolios[portfolionr].portfoliovariance**(0.5)))
    y = np.append(y, port.subportfolios[portfolionr].portfolioreturn)
    labels = np.array(port.riskreturnlabels)
    labels = np.append(labels,pf.NrClustersRiskReturn)
    print(labels)
    wlist = list()
    for i in range(port.nrassets):
        w = port.subportfolios[portfolionr].weights[i]
        if w > 0:
            w = 0.02 + math.sqrt(w)*RiskReturnPlotScaling
        wlist.append(w)           
    radii = np.array(wlist)         
    print(radii)
    radii = np.append(radii,0.1) #for portfolio data point
    print(radii)
    colors = ["#%02x%02x%02x" % (int(r), int(g), 150) \
              for r, g in zip(50+200/((pf.NrClustersRiskReturn+1)-1)*labels, \
                                   30+200//((pf.NrClustersRiskReturn+1)-1)*labels)]
    #print(colors)
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    riskreturnplot = figure(tools = TOOLS)
    #plt.scatter(self.assets[i].std,self.assets[i].growthrate, color = colors[i], \
    #            label=lbl)
    riskreturnplot.scatter(x, y, radius = radii, fill_color = colors, fill_alpha = 0.6, line_color = None)
    riskreturnplot.title.text = text
    script, div = components(riskreturnplot)
    if len(port.assets) > 0:
        nrassets = port.nrassets
    else:
        nrassets = 0
    inittext = 'Number of datasets imported from API: ' + str(nrassets)

    return render_template('apptrend.html', script = script, div = div, \
                           inittext=text)


def transformrowcovmatrix(mina, maxa, row):
    return [(element - mina)/(maxa - mina) for element in row]

@app.route('/plotcovmatrix', methods = ['GET','POST'])
def plotcovmatrix():
    # create an array of RGBA data
    N = port.nrassets
    npcovmatrix = np.array(port.subportfolios[0].covmatrix)
    print(npcovmatrix)
    mina = np.amin(npcovmatrix)
    maxa = np.amax(npcovmatrix)
    for x in np.nditer(npcovmatrix, op_flags=['readwrite']):
        x[...] = (x - mina)/(maxa - mina)
        x[...] = (x*400)**1.5 #(x*400)**1.5
    print(npcovmatrix)
    img = np.array(npcovmatrix, dtype=np.uint32)
    
    view = img.view(dtype=np.uint8).reshape((N, N, 4))
    for i in range(N):
        for j in range(N):
            view[i, j, 0] = 0 #int(npcovmatrix[i,j]*255.0) # int(255*npcovmatrix[i,j]) #was int(255*npcovmatrix[i,j]) for each
            view[i, j, 1] = 0 #100 # int(255*npcovmatrix[i,j]) 
            view[i, j, 2] = 255 #100#int(255*npcovmatrix[i,j])
            view[i, j, 3] = int(npcovmatrix[i,j]*255.0)#255 #int(255*npcovmatrix[i,j]) 

    p = figure(plot_width=600, plot_height=600, x_range=(0, N), y_range=(0, N))
    p.image_rgba(image=[img], x=[0], y=[0], dw=[N], dh=[N])
    p.title.text = 'Covariance matrix for all market indices'
    p.xaxis.axis_label = 'Market index number'
    p.yaxis.axis_label = 'Market index number'
    
    Dt = list()
    Cd = list()
    Cn = list()
    for rowi in range(len(port.subportfolios[0].covmatrix)):
        for coli in range(len(port.subportfolios[0].covmatrix[rowi])):
            Dt.append(rowi)
            Cd.append(coli)
            Cn.append(int((port.subportfolios[0].covmatrix[rowi][coli] - mina)/(maxa - mina)*10))
                          
            

 #   Dt = ['2017-01-12','2017-01-12','2017-01-12','2017-01-12','2017-01-12',
#      '2017-01-10','2017-01-10','2017-01-10','2017-01-09','2017-01-12',
#      '2017-01-11','2017-01-11','2017-01-10','2017-01-10','2017-01-09',
#      '2017-01-09','2017-01-09','2017-01-09','2017-01-11','2017-01-10',
#      '2017-01-10','2017-01-11','2017-01-11','2017-01-12','2017-01-11',
#      '2017-01-09']
 #   print('len(Dt):',len(Dt))
 #   Pr = ['GT','GT','GT','GT','GT','GT','GT','GT','GT','GT','GT','GT','GT','GT',
 #     'GT','GT','GT','GT','BW','BW','BW','BW','BW','BW', 'BW', 'BW',
 #     'GT','GT','BW','BW','BW','BW','BW','BW', 'BW', 'BW']
 #   print('len(Pr):',len(Pr))
#    Cd = [60,60,61,61,65,63,65,68,62,62,62,60,66,60,68,65,62,62,61,61,61,61,62,62,60,66]
 #   print('len(Cd):',len(Cd))
#    Cn = [1,6,0,7,7,1,2,3,8,1,2,2,4,1,1,1,1,1,0,1,6,0,1,6,5,4]
 #   print('len(Cn):',len(Cn))
 #   print('Cn:',Cn)

 #   df = pd.DataFrame({'Pr':Pr,'Dt':Dt,'Cd':Cd,'Cn':Cn})
    #df = pd.DataFrame({'Dt':Dt,'Cd':Cd,'Cn':Cn})
 #   datadict = df.to_dict(orient='list')
 #   source = ColumnDataSource(datadict)

 #   p = HeatMap(source.data, x='Cd', y='Dt', values='Cn')

    script, div = components(p)
    if len(port.assets) > 0:
        nrassets = port.nrassets
    else:
        nrassets = 0

    return render_template('plotcovmatrix.html', script = script, div = div)


@app.route('/trainnngrowth', methods = ['GET','POST'])
def trainnngrowth():
    #print(port.assets[0].pricehistory)
    #print(type(port.assets[0].pricehistory))
    port.predictreturns()
    
    # create a new plot with a title and axis labels
    p = figure(title='Assets neural network predicted growth', \
               x_axis_label='x', y_axis_label='y')

    # add a line renderer with legend and line thickness
    mypalette = viridis(port.nrassets)
    #nn predictions
    for i in RangeOfAssetsForPredictDiag:
        #print(i)
        x = np.array(range(-len(port.assets[i].fittedprediction)+1+\
                       pf.NrMonthsPredicted,1+pf.NrMonthsPredicted))
        p.line(x, \
               port.assets[i].fittedprediction, legend="Predicted " + port.assets[i].name, \
               line_color = mypalette[i], \
               line_width=2)

    #nn output actuals
    for i in RangeOfAssetsForPredictDiag:
        #print(i)
        x = np.array(range(-len(port.assets[i].pricehistory)+1,1))
        p.line(x, \
               port.assets[i].pricehistory, \
               legend="Actual " + port.assets[i].name, \
               line_color = mypalette[i], \
               line_width=1)
        
    p.legend.label_text_font_size = '8pt'
    p.legend.location = "top_left"

    # show the results
    plot = p
    script, div = components(plot)
    inittext = 'Number of datasets imported from API: ' + str(len(port.assets))
    return render_template('apptrend.html', script = script, div = div, \
                           inittext=inittext)


@app.route('/oneassetdetail', methods = ['GET','POST'])
def oneassetdetail():

    port.predictreturns()
    inittext = 'Growth rate: ' + str(round(port.assets[AssetChosenForDetails].growthpyearmean,3)) + \
               '; Risk index: ' + str(round(port.assets[AssetChosenForDetails].growthpyearstd,3))
    p = figure(title='Asset details for '+port.assets[AssetChosenForDetails].name + ': ' + inittext, \
               x_axis_label='x', y_axis_label='y')

    mypalette = viridis(port.nrassets)
 
    #fitted prediction plot
    x = np.array(range(-len(port.assets[AssetChosenForDetails].fittedprediction)+1+\
                       pf.NrMonthsPredicted,1+pf.NrMonthsPredicted))
    p.line(x, \
               port.assets[AssetChosenForDetails].fittedprediction, legend="Predicted " + \
           port.assets[AssetChosenForDetails].name, \
               line_color = mypalette[0], \
               line_width=2)

    #actual price history plot
    x = np.array(range(-len(port.assets[AssetChosenForDetails].pricehistory)+1,1))
    p.line(x, \
               port.assets[AssetChosenForDetails].pricehistory, \
               legend="Actual " + port.assets[AssetChosenForDetails].name, \
               line_color = mypalette[0], \
               line_width=1)
        
    p.legend.label_text_font_size = '8pt'
    p.legend.location = "top_left"

    # show the results
    plot = p
    script, div = components(plot)
    
    return render_template('apptrend.html', script = script, div = div, \
                           inittext=inittext)


@app.route('/investigatehyperspace', methods = ['GET','POST'])
def investigatehyperspace():
    inittext = 'Investigate hyperspace to be optimized'
    resolution = 20
    N = resolution

    startw = list()
    for i in range(port.nrassets):
        startw.append(random.random())
    asset1 = random.randint(0, port.nrassets - 1)
    asset2 = asset1
    while asset2 == asset1:
        asset2 = random.randint(0, port.nrassets - 1)

    
    hyperspace = np.zeros(((resolution), (resolution)))
    for i in range(resolution): 
        for j in range(resolution): 
            startw[asset1] = i/float(resolution)
            startw[asset2] = j/float(resolution)
            hyperspace[i, j] = port.subportfolios[1].funcriskreturn(startw)
    
    npcovmatrix = np.array(hyperspace)

    
    print(npcovmatrix)
    mina = np.amin(npcovmatrix)
    maxa = np.amax(npcovmatrix)
    for x in np.nditer(npcovmatrix, op_flags=['readwrite']):
        x[...] = (x - mina)/(maxa - mina)
        x[...] = (x*400)**1.5
    print(npcovmatrix)
    img = np.array(npcovmatrix, dtype=np.uint32)
    
    view = img.view(dtype=np.uint8).reshape((N, N, 4))
    for i in range(N):
        for j in range(N):
            view[i, j, 0] = int(255*npcovmatrix[i,j]) #was 255 for each
            view[i, j, 1] = int(255*npcovmatrix[i,j]) 
            view[i, j, 2] = int(255*npcovmatrix[i,j])
            view[i, j, 3] = int(255*npcovmatrix[i,j]) 

    p = figure(plot_width=400, plot_height=400, x_range=(0, N), y_range=(0, N))
    p.image_rgba(image=[img], x=[0], y=[0], dw=[N], dh=[N])
    Dt = list()
    Cd = list()
    Cn = list()
    for rowi in range(len(port.subportfolios[0].covmatrix)):
        for coli in range(len(port.subportfolios[0].covmatrix[rowi])):
            Dt.append(rowi)
            Cd.append(coli)
            Cn.append(int((port.subportfolios[0].covmatrix[rowi][coli] - mina)/(maxa - mina)*10))
    
    script, div = components(p)
    if len(port.assets) > 0:
        nrassets = port.nrassets
    else:
        nrassets = 0
    
    return render_template('apptrend.html', script = script, div = div, \
                           inittext=inittext)
    
    

@app.route('/optimiseportfolio', methods = ['GET','POST'])
def optimiseportfolio():
    port.optimiseportfolio()
    return render_template('appindex.html', terminaltext = port.terminaltext)


@app.route('/portwouldbe', methods = ['GET','POST'])
def portwouldbe():
    port.wouldbetrendlist = list()
    for i in range(len(port.subportfolios)):
        wouldbetrendlist.append(port.howportwouldhavedone(i))
    dill.dump(wouldbetrendlist, open('wouldbetrend.pkd', 'wb'))

    portfolionr = 0
    if request.method == 'POST':
        if request.form['submit'] == 'Risk Level 0':
            portfolionr = 0
        elif request.form['submit'] == 'Risk Level 1':
            portfolionr = 1
        elif request.form['submit'] == 'Risk Level 2':
            portfolionr = 2
        elif request.form['submit'] == 'Risk Level 3':
            portfolionr = 3
        elif request.form['submit'] == 'Risk Level 4':
            portfolionr = 4
        elif request.form['submit'] == 'Risk Level 5':
            portfolionr = 5
        elif request.form['submit'] == 'Risk Level 6':
            portfolionr = 6
 

    mypalette = viridis(len(port.subportfolios))
 
    inittext = 'What return would have been had you chosen IIPM ' + str(pf.NumMonthsMonthsPreT0Fit) + ' months ago and invested.'
    p = figure(title=inittext, x_axis_label='x', y_axis_label='y')
    x = np.array(range(pf.NumMonthsMonthsPreT0Fit + 1))

    for i in range(len(port.subportfolios)):
        p.line(x, wouldbetrendlist[i]*100.0, legend= "Risk level " + str(i) , line_color = mypalette[i], \
               line_width=2)

    #p.legend.label_text_font_size = '8pt'
    p.legend.location = "top_left"

    script, div = components(p)
    
    return render_template('apptrend.html', script = script, div = div, \
                           inittext=inittext)


@app.route('/portwouldbeloaded', methods = ['GET','POST'])
def portwouldbeloaded():
    
    nrmonthspreT0fit = len(port.wouldbetrendlist[0]) - 1
    mypalette = viridis(len(port.subportfolios))
 
    inittext = 'What return would have been had you chosen IIPM ' + str(nrmonthspreT0fit) + ' months ago and invested.'
    p = figure(title=inittext, x_axis_label='x', y_axis_label='y')
    x = np.array(range(nrmonthspreT0fit + 1))

    for i in range(1, len(port.subportfolios)):
        p.line(x, port.wouldbetrendlist[i]*100.0, legend= "Risk level " + str(i) , line_color = mypalette[i - 1], \
               line_width=2)

    #p.legend.label_text_font_size = '8pt'
    p.legend.location = "top_left"
    p.xaxis.axis_label = 'Time (months to T0)'
    p.yaxis.axis_label = 'Result of USD 100 invested (USD)'

    script, div = components(p)
    
    return render_template('portwouldbeloaded.html', script = script, div = div, \
                           inittext=inittext)






@app.route('/viewsubportfolios', methods = ['GET','POST'])
def viewsubportfolios():
    port.optimiseportfolio(False)

    portfolionr = 0
    if request.method == 'POST':
        portfolionr = int(request.form['pageinput'])

    #output_file("bar_colormapped.html")

    counts = list()
    fruits = list()
    y = list()
    j = 0
    for i in range(port.nrassets):
        w = port.subportfolios[portfolionr].weights[i]
        if w > 0:
            j += 1
            y.append(j)
            w = w*100 #0.02 + math.sqrt(w)*RiskReturnPlotScaling
            counts.append(w)        
            fruits.append(port.subportfolios[portfolionr].assets[i].name)



    #radii = np.array(wlist)  

    #fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
    #counts = [5, 3, 4, 2, 4, 6]

    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

    #p = figure(y_range=fruits, plot_width=350, toolbar_location=None, title="Fruit Counts")
    
    #p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
     #     line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))
    
    #p = figure(plot_width=600, plot_height=600)
    #p.hbar(y=fruits, height=0.5, left=0,
     #  right=counts, color="navy")


    #p = figure(plot_width=400, plot_height=400)
    p = figure(y_range = fruits)
    p.hbar(y='fruits', height=0.5, left=0,
       right=counts ,source=source,
       line_color='white', color='navy')
    #p.hbar(y='fruits', right='counts', height=0.9, left = 0, source=source, legend="fruits",
       #   line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

    #p.xgrid.grid_line_color = None
    #p.y_range.start = 0
    #p.y_range.end = 9
    p.x_range.start = 0
    p.x_range.end = 100
    #p.legend.orientation = "vertical"
    #p.legend.location = "bottom_right"

    script, div = components(p)
    
    return render_template('apptrend.html', script = script, div = div, \
                           inittext='')

    #return render_template('appindex.html', terminaltext = port.terminaltext)


@app.route('/viewsubportfoliostext', methods = ['GET','POST']) #this function will trend the time series data for all assets
def viewsubportfoliostext(): #remember the function name does not need to match the URL
    port.terminaltext = ''
    port.optimiseportfolio(False)
    return render_template('allportfolios.html', terminaltext = port.terminaltext)


@app.route('/viewportfolioriskreturn', methods = ['GET','POST']) #this function will trend the time series data for all assets
def viewportfolioriskreturn(): #remember the function name does not need to match the URL
    port.terminaltext = ''
    for i in range(port.nrassets):
        s = port.assets[i].name
        if len(s)//8 <= 1:
            s += '\t\t'
        else:
            s+= '\t'
        s += str(port.assets[i].growthpyearmean)
        s += '\t' + str(port.assets[i].growthpyearstd)
        port.printtoterminal(s)

    return render_template('allportfolios.html', terminaltext = port.terminaltext)

@app.route('/pitch0', methods = ['GET','POST']) 
def pitch0(): 
    return render_template('pitch0.html', terminaltext = '')

@app.route('/pitch1', methods = ['GET','POST']) 
def pitch1(): 
    return render_template('pitch1.html', terminaltext = '')

@app.route('/pitch2', methods = ['GET','POST']) 
def pitch2(): 
    return render_template('pitch2.html', terminaltext = '')


if __name__ == '__main__':

    app.run(port=33507, debug=False)





