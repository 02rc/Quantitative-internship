import xlrd
import math
import numpy
from scipy.optimize import minimize


######## functions definitions ###############################################


def SABR(alpha,beta,rho,nu,F,K,time,MKT): # all variables are scalars

    if K <= 0:   # negative rates' problem, need to shift the smile
        VOL = 0
        diff = 0
    elif F == K: # ATM formula
        V = (F*K)**((1-beta)/2.)
        logFK = math.log(F/K)
        A = 1 + ( ((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2-3*(rho**2))/24.) ) * time
        B = 1 + (1/24.)*(((1-beta)*logFK)**2) + (1/1920.)*(((1-beta)*logFK)**4)
        VOL = (alpha/V)*A
        diff = VOL - MKT
    elif F != K: # not-ATM formula
        V = (F*K)**((1-beta)/2.)
        logFK = math.log(F/K)
        z = (nu/alpha)*V*logFK
        x = math.log( ( math.sqrt(1-2*rho*z+z**2) + z - rho ) / (1-rho) )
        A = 1 + ( ((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2-3*(rho**2))/24.) ) * time
        B = 1 + (1/24.)*(((1-beta)*logFK)**2) + (1/1920.)*(((1-beta)*logFK)**4)
        VOL = (nu*logFK*A)/(x*B)
        diff = VOL - MKT

    print round(VOL,4) ,  '\t' ,
    outvol.write('%r;' %round(VOL,4) )
    if MKT==0:
        diff = 0
        vol_diff.write('%s;' %'No market data')
    else:
        vol_diff.write('%r;' %round(diff,4) )


def smile(alpha,beta,rho,nu,F,K,time,MKT,i): # F, time and the parameters are scalars, K and MKT are vectors, i is the index for tenor/expiry label

    print label_ten[i] , '\t' , label_exp[i] , '\t' ,
    outvol.write('%s;%s;' %(label_ten[i],label_exp[i]))
    vol_diff.write('%s;%s;' %(label_ten[i],label_exp[i]))
    parameters.write('%s;%s;' %(label_ten[i],label_exp[i]))

    for j in range(len(K)):
        if K[0] <= 0:
            shift(F,K)
        SABR(alpha,beta,rho,nu,F,K[j],time,MKT[j])

    print ' '
    outvol.write('\n')
    vol_diff.write('\n')
    parameters.write('%f;%f;%f;%f;' %(alpha ,beta ,rho ,nu))
    parameters.write('\n')


def SABR_vol_matrix(alpha,beta,rho,nu,F,K,time,MKT): # F, time and the parameters are vectors, K and MKT are matrices

    print ' '
    print (2+((num_strikes-1)/2))*'       '+'SABR VOLATILITIES'
    print '  ' , '\t' , 'strikes:' , 
    for i in range(num_strikes):
        print label_strikes[i] , '\t' ,
    print ' '
    outvol.write('%s;' %'SABR VOLATILITIES')
    outvol.write('\n')
    vol_diff.write('%s;' %'VOLATILITY DIFFERENCES')
    vol_diff.write('\n')
    parameters.write('%s;' %'PARAMETERS')
    parameters.write('\n')
    outvol.write('%s;%s;' %(' ','strikes:'))
    vol_diff.write('%s;%s;' %(' ','strikes:'))
    for j in range(len(strike_spreads)):
        outvol.write('%s;' %label_strikes[j])
        vol_diff.write('%s;' %label_strikes[j])
    outvol.write('\n')
    vol_diff.write('\n')
    print 'tenor' , '\t' ,   'expiry'
    parameters.write('%s;%s;%s;%s;%s;%s' %('tenor','expiry','alpha','beta','rho','nu'))
    parameters.write('\n')

    for i in range(len(F)):
        smile(alpha[i],beta[i],rho[i],nu[i],F[i],K[i],time[i],MKT[i],i)



def shift(F,K):
    shift = 0.001 - K[0]
    for j in range(len(K)):
        K[j] = K[j] + shift
        F = F + shift   



def objfunc(par,F,K,time,MKT):
    sum_sq_diff = 0
    if K[0]<=0:
        shift(F,K)
    for j in range(len(K)):
        if MKT[j] == 0:   
            diff = 0       
        elif F == K[j]: 
            V = (F*K[j])**((1-par[1])/2.)
            logFK = math.log(F/K[j])
            A = 1 + ( ((1-par[1])**2*par[0]**2)/(24.*(V**2)) + (par[0]*par[1]*par[3]*par[2])/(4.*V) + ((par[3]**2)*(2-3*(par[2]**2))/24.) ) * time
            B = 1 + (1/24.)*(((1-par[1])*logFK)**2) + (1/1920.)*(((1-par[1])*logFK)**4)
            VOL = (par[0]/V)*A
            diff = VOL - MKT[j]
        elif F != K[j]: 
            V = (F*K[j])**((1-par[1])/2.)
            logFK = math.log(F/K[j])
            z = (par[3]/par[0])*V*logFK
            x = math.log( ( math.sqrt(1-2*par[2]*z+z**2) + z - par[2] ) / (1-par[2]) )
            A = 1 + ( ((1-par[1])**2*par[0]**2)/(24.*(V**2)) + (par[0]*par[1]*par[3]*par[2])/(4.*V) + ((par[3]**2)*(2-3*(par[2]**2))/24.) ) * time
            B = 1 + (1/24.)*(((1-par[1])*logFK)**2) + (1/1920.)*(((1-par[1])*logFK)**4)
            VOL = (par[3]*logFK*A)/(x*B)
            diff = VOL - MKT[j]  
        sum_sq_diff = sum_sq_diff + diff**2  
        obj = math.sqrt(sum_sq_diff)
    return obj


def calibration(starting_par,F,K,time,MKT):
    for i in range(len(F)):
        x0 = starting_par
        bnds = ( (0.001,None) , (0,1) , (-0.999,0.999) , (0.001,None)  )
        res = minimize(objfunc, x0 , (F[i],K[i],time[i],MKT[i]) ,bounds = bnds, method='SLSQP') # for a constrained minimization of multivariate scalar functions
        alpha[i] = res.x[0]
        beta[i] = res.x[1]
        rho[i] = res.x[2]
        nu[i] = res.x[3]


################################################################################################################################################


######## inputs and outputs #########################################

outvol = open('outvol.csv', 'w')             # file output of volatilities
vol_diff = open('vol differences.csv', 'w')  # file output differences between SABR and Market volatilities
parameters = open('parameters.csv', 'w')     # file output parameters


while True:
    try:
        file_input = xlrd.open_workbook('market_data.xlsx')     # load market data
    except:
        print 'Input file is not in the directory!'
    break
Market_data = file_input.sheet_by_name('Swaptions data')        # file input forward rates


######## set swaptions characteristics ###############################
     
strike_spreads=[]
j=0
while True:
    try:
        strike_spreads.append(int(Market_data.cell(1,3+j).value))
        j = j+1
    except:
        break
num_strikes = len(strike_spreads)

expiries=[]
i=0
while True:
        try:
            expiries.append(Market_data.cell(2+i,1).value)
            i = i + 1
        except:
            break

tenors=[]
i=0
while True:
    try:
        tenors.append(Market_data.cell(2+i,0).value)
        i = i + 1
    except:
        break


# to create the ATM forward rates
F = []
i=0
while True:
    try:
        F.append(Market_data.cell(2+i,2).value)
        i = i+1
    except:
        break

# to create the strike grid
K = numpy.zeros((len(F),num_strikes))
for i in range(len(F)):
    for j in range(num_strikes):
        K[i][j] = F[i] + 0.0001*(strike_spreads[j])  

# to create market volatilities            
MKT = numpy.zeros((len(F),num_strikes))
for i in range(len(F)):
    for j in range(num_strikes):
        MKT[i][j] = Market_data.cell(2+i,3+j).value


# set starting parameters
starting_guess = numpy.array([0.001,0.5,0,0.001])
alpha = len(F)*[starting_guess[0]]
beta = len(F)*[starting_guess[1]]
rho = len(F)*[starting_guess[2]]
nu = len(F)*[starting_guess[3]]


######## set labels ###################################################

exp_dates = len(expiries)*[0]
for i in range(len(expiries)):
    if expiries[i] < 1:
        exp_dates[i] = str(int(round(12*expiries[i])))+'m'
    else:
        exp_dates[i] = str(int(round(expiries[i])))+'y'
        if expiries[i]-round(expiries[i]) > 0:
            exp_dates[i] = exp_dates[i]+str(int(round((12*(round(expiries[i],2)-int(expiries[i]))))))+'m' 
        elif expiries[i]-round(expiries[i]) < 0:
            exp_dates[i] = str(int(round(tenors[i]))-1)+'y'
            exp_dates[i] = exp_dates[i]+str(int(round((12*(round(expiries[i],2)-int(expiries[i]))))))+'m'

ten_dates = len(tenors)*[0]
for i in range(len(tenors)):
    if tenors[i] < 1:
        ten_dates[i] = str(int(round(12*tenors[i])))+'m'
    else:
        ten_dates[i] = str(int(round(tenors[i])))+'y'
        if tenors[i]-round(tenors[i]) > 0:
            ten_dates[i] = ten_dates[i]+str(int(round((12*(round(tenors[i],2)-int(tenors[i]))))))+'m' 
        elif tenors[i]-round(tenors[i]) < 0:
            ten_dates[i] = str(int(round(tenors[i]))-1)+'y'
            ten_dates[i] = ten_dates[i]+str(int(round((12*(round(tenors[i],2)-int(tenors[i]))))))+'m'

label_exp = exp_dates
label_ten = ten_dates
label_strikes = num_strikes*[0]
for i in range(num_strikes):
    if strike_spreads[i] == 0 :
        label_strikes[i] = 'ATM'
    else:
        label_strikes[i] = str(strike_spreads[i])


######## Call the functions #################################

calibration(starting_guess,F,K,expiries,MKT)

SABR_vol_matrix(alpha,beta,rho,nu,F,K,expiries,MKT)



######## Close output files #################################

outvol.close()
vol_diff.close()
parameters.close()
