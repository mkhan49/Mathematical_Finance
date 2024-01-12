import numpy as np
import scipy.stats

class Barrier_Options:

    def __init__(self, S0, K, T, r, mu, B, Bp, sigma, NSteps, NRepl, NPilot):

        '''
        Parameters
        ==========
        S0 : float
            stock/index level at t0
        K : float
            strike price
        T : float
            Number of years till maturity
        r : float
            constant, risk-free short rate
        mu : float
            constant, mean
        B : float
            barrier price
        Bp : float
            barrier price
        sigma : float
            volatility
        NSteps : int
            constant, number of discrete steps  of T
        NRepl : int
            constant, number of replications of stock paths
        NPilot : int


        '''
        self.S0 = S0
        self.K  = K
        self.T  = T
        self.r  = r
        self.mu = mu
        self.B  = B
        self.Bp = Bp
        self.sigma  = sigma
        self.NSteps = NSteps
        self.NRepl  = NRepl
        self.NPilot = NPilot

    def BSM_Call_Value(self):
        ''' Calculates Black-Scholes-Merton European call option value.

        Parameters
        ==========
        S0, K, T, r, sigma 

        Returns
        =======
        call_value : float
            European call present value at initial time
        '''  
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        call_value = self.S0 * scipy.stats.norm.cdf(d1,0,1) - np.exp(-self.r * self.T )* self.K * scipy.stats.norm.cdf(d2,0,1)
        
        return call_value


    def Up_Out_Call(self):         
        ''' Calculates Black-Scholes-Merton Up and out barrier call option value.

        Parameters
        ==========
        S0, K, B, T, r, sigma

        Returns
        =======
        call_value : float
            Up and out barrier call present value at initial time
        '''    
        if self.K > self.B:
            raise ValueError ("Barrier value cannot be less than strike price.")

    
        a = (self.S0/self.B) ** (- 2 * self.r / self.sigma ** 2)
        b = (self.S0/self.B) ** (1 - (2 * self.r / self.sigma ** 2))
        
        d1 = ( np.log(self.S0/self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))
        d2 = ( np.log(self.S0/self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))
        d3 = ( np.log(self.S0/self.B) + (self.r + 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))    
        d4 = ( np.log(self.S0/self.B) + (self.r - 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))    
        d5 = ( np.log(self.B/self.S0) + (self.r + 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))    
        d6 = ( np.log(self.B/self.S0) + (self.r - 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))    
        d7 = ( np.log(self.B ** 2/(self.S0 * self.K)) + (self.r + 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))    
        d8 = ( np.log(self.B ** 2/(self.S0 * self.K)) + (self.r - 0.5 * self.sigma ** 2) * self.T ) / (self.sigma * np.sqrt(self.T))       
        
        p = self.S0 * (scipy.stats.norm.cdf(d1,0,1) - scipy.stats.norm.cdf(d3,0,1)) - self.K * np.exp(-self.r * self.T) * (scipy.stats.norm.cdf(d2,0,1) - scipy.stats.norm.cdf(d4,0,1)) - self.B * a * (scipy.stats.norm.cdf(d7,0,1) - scipy.stats.norm.cdf(d5,0,1)) + self.K * np.exp(-self.r * self.T) * b * (scipy.stats.norm.cdf(d8,0,1) - scipy.stats.norm.cdf(d6,0,1))

        return p


    def Stock_Paths(self):  
        ''' Generates the stock paths for different replications

        Parameters
        ==========
        S0, mu, sigma, T, NSteps, NRepl

        Returns
        =======
        spaths :vector
            Replications of stock paths for discrete steps of T 
        '''
        spaths = np.zeros((self.NRepl,1+self.NSteps))
        spaths[:,0] = self.S0
        dt = self.T/self.NSteps
        nudt = (self.mu- 0.5 * self.sigma ** 2) * dt
        sidt = self.sigma * np.sqrt(dt)
        for i in range(0,self.NRepl):
            for j  in range(1,self.NSteps+1):
                spaths[i,j] = spaths[i,j-1] * (np.exp(nudt + (sidt * np.random.normal(0,1))))
        return spaths


    def Stock_Paths_AV(self):
        ''' Generates the two different stock paths for different replications

        Parameters
        ==========
        S0, mu, sigma, T, NSteps, NRepl
        
        Returns
        =======
        spaths_a, spaths_b :vector
            Replications of two different stock paths for discrete steps of T 
        '''
        spaths_a = np.zeros((self.NRepl,1+self.NSteps))
        spaths_a[:,0] = self.S0
        spaths_b = np.zeros((self.NRepl,1+self.NSteps))
        spaths_b[:,0] = self.S0
        dt = self.T/self.NSteps
        nudt = (self.mu- 0.5 * self.sigma ** 2) * dt
        sidt = self.sigma * np.sqrt(dt)
        for i in range(0,self.NRepl):
            for j  in range(1,self.NSteps+1):
                spaths_a[i,j] = spaths_a[i,j-1] * (np.exp(nudt + (sidt * np.random.normal(0,1))))         
                spaths_b[i,j] = spaths_b[i,j-1] * (np.exp(nudt + (sidt * np.random.normal(0,1))))
        return spaths_a, spaths_b 
   
     
    def Up_Out_Call_Monte_Carlo(self, NRepl):
        ''' Calculates the Up and out barrier call value using monte carlo simulations

        Parameters
        ==========
        S0, K, r, T, sigma, B, NSteps, NRepl

        Returns
        =======
        mean, std error :float
            Option price and standard error 
        '''
        if self.K > self.B:
            raise ValueError ("Barrier value cannot be less than strike price.")
            
        payoff = np.zeros((self.NRepl,1))
        ncrossed = 0
        for i in range(1,self.NRepl):
            self.mu = self.r
            self.NRepl = 1
            path = self.Stock_Paths()
            crossed = path >= self.B
            self.NRepl = NRepl
            
            if crossed.sum() == 0 :
                payoff[i,0] = max(0, path[:,self.NSteps] - self.K)
            else:
                payoff[i,0] = 0
                ncrossed = ncrossed + 1
            
        mean, std = scipy.stats.norm.fit(np.exp(-self.r * self.T) * payoff)
        return mean, std


    def Up_Out_Call_Antithetic(self, NRepl):
        ''' Calculates the Up and out barrier call value using 
        monte carlo simulation and antithetic variates variance reduction technique

        Parameters
        ==========
        S0, B, K, r, T, sigma, NSteps, NRepl

        Returns
        =======
        mean, std error :float
            Option price and standard error 
        '''
        if self.K > self.B:
            raise ValueError ("Barrier value cannot be less than strike price.")
            
        payoff_a = np.zeros((self.NRepl,1))
        payoff_b = np.zeros((self.NRepl,1))
        
        for i in range(1,self.NRepl):
            self.mu = self.r
            self.NRepl = 1
            path_a, path_b = self.Stock_Paths_AV()
            self.NRepl = NRepl
            
            crossed_a = path_a >= self.B
            
            if crossed_a.sum() == 0 :
                payoff_a[i,0] = max(0, path_a[:,self.NSteps] - self.K)
            crossed_b = path_b >= self.B
            
            if crossed_b.sum() == 0 :
                payoff_b[i,0] = max(0, path_b[:,self.NSteps] - self.K)
                
        payoff = (payoff_a + payoff_b) / 2
        mean, std = scipy.stats.norm.fit(np.exp(-self.r * self.T) * payoff)  
        return mean, std
    

    def Up_Out_Call_Control_Variate(self, NRepl):
        ''' Calculates the Up and out barrier call value using 
        monte carlo simulation and control variates variance reduction technique

        Parameters
        ==========
        S0, B, K, r, T, sigma, NSteps, NRepl, NPilot

        Returns
        =======
        mean, std error :float
            Option price and standard error 
        '''   
        if self.K > self.B:
            raise ValueError ("Barrier value cannot be less than strike price.")
            
        call_value = self.BSM_Call_Value()
        
        #Vanilla Payoff variables
        
        payoff = np.zeros((self.NPilot,1))
        vanilla_payoff = np.zeros((self.NPilot,1))
        
        for i in range(0,self.NPilot):
            self.mu = self.r
            self.NRepl = 1
            path = self.Stock_Paths()
            vanilla_payoff[i,:] = max(0, path[:,self.NSteps] - self.K)
            crossed = path >= self.B
            
            if crossed.sum() == 0 :
                payoff[i,:] = max(0, path[:,self.NSteps] - self.K)
                
        vanilla_payoff = np.exp(-self.r * self.T) * vanilla_payoff
        payoff = np.exp(-self.r * self.T) * payoff
        
        covar_vanilla = np.cov(vanilla_payoff,payoff,bias=True)
        var_vanilla = np.var(vanilla_payoff)
        corr = - covar_vanilla[0,1] / var_vanilla
        
        #New Payoff variables
        
        self.NRepl = NRepl
        new_payoff = np.zeros((self.NRepl,1))
        new_vanilla_payoff = np.zeros((self.NRepl,1))
        for i in range(0,self.NRepl):
            self.mu = self.r
            self.NRepl = 1
            path = self.Stock_Paths()
            self.NRepl = NRepl
            new_vanilla_payoff[i,:] = max(0, path[:,self.NSteps] - self.K)
            crossed = path >= self.B
            
            if crossed.sum() == 0 :
                new_payoff[i] = max(0, path[:,self.NSteps] - self.K)
        
        new_vanilla_payoff = np.exp(-self.r * self.T) * new_vanilla_payoff
        new_payoff  = np.exp(-self.r * self.T) * new_payoff
        cv_payoff  = new_payoff + (corr * (new_vanilla_payoff - call_value))
        mean, std = scipy.stats.norm.fit(np.exp(-self.r * self.T) * cv_payoff)
        
        return mean, std
    

    def Up_Out_Call_Conditional_Monte_Carlo(self, NRepl, T):
        ''' Calculates the Up and out barrier call value using 
        monte carlo simulation and conditional expectations variance reduction technique

        Parameters
        ==========
        S0, B, K, r, T, sigma, NSteps, NRepl

        Returns
        =======
        mean, std error :float
        Option price and standard error 
        '''   
        if self.K > self.B:
            raise ValueError ("Barrier value cannot be less than strike price.")
            
        call_value = self.BSM_Call_Value()
        
        dt = self.T/self.NSteps
        ncrossed = 0
        payoff = np.zeros((self.NRepl,1))
        times = np.zeros((self.NRepl,1))
        stock_values = np.zeros((self.NRepl,1))
        call_value_new = np.zeros((self.NRepl,1))
        
        for i in range(1,self.NRepl):
            self.mu = self.r
            self.NRepl = 1
            path = self.Stock_Paths()
            self.NRepl = NRepl
            a,b = np.where(path >= self.B)
            if len(b) == 0 :
                tcrossed = 0
            else:
                tcrossed = min(b)
                ncrossed = ncrossed + 1
                times[ncrossed-1,:] = (tcrossed - 1) * dt
                stock_values[ncrossed-1,:] = path[:,tcrossed]

        if ncrossed > 0:
            for j in range(0,ncrossed):
                self.S0 = stock_values[j,0]
                self.T  = self.T-times[j,0]
                call_value_new[j,0] = self.BSM_Call_Value()
                payoff[j,0] =  np.exp(-self.r * times[j,0]) * call_value_new[j,0]
                self.T = T
                          
        mean, std = scipy.stats.norm.fit(call_value - payoff)
        return mean, std 


    def Up_Out_Call_Importance_Sampling(self, S0, T):
        ''' Calculates the Up and out barrier call value using 
        monte carlo simulation and importance sampling variance reduction technique

        Parameters
        ==========
        S0, B, K, r, T, sigma, NSteps, NRepl, Bp


        Returns
        =======
        mean, std error :float
        Option price and standard error 
        '''   
        if self.K > self.B:
            raise ValueError ("Barrier value cannot be less than strike price.")
            
        dt = self.T/self.NSteps
        nudt = (self.r - 0.5 * self.sigma ** 2) * dt
        q = self.Bp * nudt
        sidt = self.sigma * np.sqrt(dt)
        
        call_value = self.BSM_Call_Value()
        
        ncrossed = 0
        payoff = np.zeros((self.NRepl,1))
        times = np.zeros((self.NRepl,1))
        stock_values = np.zeros((self.NRepl,1))
        is_ratio = np.zeros((self.NRepl,1))
        call_value_new = np.zeros((self.NRepl,1))
        vetz = np.zeros((1,self.NSteps))
        
        
        for i in range(1,self.NRepl):
            for j in range(0,self.NSteps):
                vetz[0,j]= nudt - q + (sidt * np.random.normal(0,1))
            x = np.zeros((1,self.NSteps+1))
            x[:,0] = np.log(self.S0)
            x[:,1:self.NSteps+1] = vetz[:,0:self.NSteps]
            path = np.exp(np.cumsum(x))
            b = min(np.where(path >= self.B))
            
            if len(b) == 0 :
                jcrossed = 0
            else:
                jcrossed = min(b)
                ncrossed = ncrossed + 1
                tbreach = jcrossed - 1
                times[ncrossed-1,:] = tbreach * dt
                stock_values[ncrossed-1,:] = path[jcrossed]
                is_ratio[ncrossed-1,:] = np.exp(((tbreach * q ** 2) / (2 * self.sigma ** 2 * dt)) + ((q / self.sigma ** 2 * dt) * np.sum(vetz[1:tbreach])) - ((tbreach * q  * (self.r - (self.sigma ** 2 / 2))) / self.sigma ** 2))     

        if ncrossed > 0:
            for k in range(0,ncrossed):
                self.S0 = stock_values[k,0]
                self.T = self.T-times[k,0]
                call_value_new[k,0] = self.BSM_Call_Value()
                payoff[k,0] =  np.exp(-self.r * times[k,0]) * call_value_new[k,0] * is_ratio[k,0]
                self.T = T
                self.S0 = S0

        mean, std = scipy.stats.norm.fit(call_value-payoff)
        return mean, std
        